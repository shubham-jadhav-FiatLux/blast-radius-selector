"""Chooses the sqlglot tests a patch can break and splits them across workers.

Input:  /data/tests.json and /data/commits/<id>.patch
Output: one JSON line per commit, {"commit": id, "workers": [[test numbers], ...]}

Everything slow happens ahead of time: ``profiling/`` runs the sqlglot suite at the pinned
commit and records what every test executes and reads, and the pinned sources are indexed
line by line. At run time each edit is classified and answered from that map:

* comment or blank-line edits outside string literals change nothing;
* an edit inside a function body is charged to every test that ran the enclosing statement;
* flipping an ``arg_types`` flag is charged to the tests that validated a node of an
  affected class while that argument was missing (that flag has no other effect);
* other module or class level edits are charged to every test that can import the module;
* data files are charged to the tests that opened them;
* anything the map cannot vouch for, including edits that no longer compile, runs everything.

Standard library only. Logs go to stderr; stdout carries nothing but answers.
"""

from __future__ import annotations

import heapq
import json
import marshal
import os
import re
import sys
import zlib

DATA = os.environ.get("BLAST_DATA", "/data")
HERE = os.path.dirname(os.path.abspath(__file__))
MAP_PATH = os.environ.get("BLAST_MAP", os.path.join(HERE, "map.bin"))
SOURCE_ROOT = os.environ.get("BLAST_SOURCE", os.path.join(HERE, "sqlglot-src"))

NO_TESTS, GLOBAL = -1, -2
NONE, RUNTIME, IMPORT, MIXED, DOC = 0, 1, 2, 3, 4
IN_STRING, STRING_AFTER, CONTINUED = 1, 2, 4
WORKER_MS = 2000
MAX_WORKERS = 8
PER_QUERY_BUDGET = 12.0  # seconds of re-checking worth trying instead of a 21 s test

CONFIG_FILES = {"pyproject.toml", "setup.cfg", "tox.ini", "pytest.ini", "conftest.py", ".coveragerc"}
CODING_RE = re.compile(r"^[ \t\f]*#.*?coding[:=]")
SIMPLE_VALUE_RE = re.compile(r"True|False|None|[-+]?\d+")
ASSIGN_TARGET_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?::[^=]*)?=(?!=)")
IMPORT_BASELINE = os.environ.get("BLAST_IMPORT_BASELINE", os.path.join(HERE, "import_baseline.txt"))

# Imports the given sqlglot modules with some files replaced by patched copies, and prints the
# ones that fail. argv: <pinned sources> <overlay dir> <modules to skip> <modules to import>
IMPORT_CHECK = r"""
import importlib, importlib.util, os, sys
root, overlay = sys.argv[1], sys.argv[2]
skip = set(filter(None, sys.argv[3].split(",")))
wanted = list(filter(None, sys.argv[4].split(",")))
patched = {}
for base, _dirs, names in os.walk(overlay):
    for name in names:
        full = os.path.join(base, name)
        rel = os.path.relpath(full, overlay)
        mod = rel[:-3].replace(os.sep, ".")
        if mod.endswith(".__init__"):
            patched[mod[:-9]] = (full, [os.path.join(root, os.path.dirname(rel))])
        else:
            patched[mod] = (full, None)
class Overlay:
    @staticmethod
    def find_spec(name, path=None, target=None):
        hit = patched.get(name)
        if hit is None:
            return None
        return importlib.util.spec_from_file_location(name, hit[0], submodule_search_locations=hit[1])
sys.meta_path.insert(0, Overlay)
sys.path.insert(0, root)
failed = []
for name in ["sqlglot"] + wanted:
    if name in skip:
        continue
    try:
        importlib.import_module(name)
    except BaseException:
        failed.append(name)
print(",".join(failed))
"""


def log(*parts):
    print("blast:", *parts, file=sys.stderr)


# ----------------------------------------------------------------------------- patches
class Block:
    """One run of removed/added lines. ``start`` is the old line number of the first removed
    line; for a pure insertion it is the old line the new lines are inserted before."""

    __slots__ = ("start", "old", "new")

    def __init__(self, start):
        self.start = start
        self.old = []
        self.new = []


class FileChange:
    __slots__ = ("path", "old_path", "status", "blocks")

    def __init__(self, old_path, path):
        self.old_path = old_path
        self.path = path
        self.status = "modify"
        self.blocks = []


_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_DIFF = re.compile(r"^diff --git a/(.*) b/(.*)$")


def parse_patch(text):
    changes = []
    cur = None
    block = None
    old_left = new_left = 0
    old_ln = 0
    for line in text.split("\n"):
        if old_left > 0 or new_left > 0:
            tag = line[:1]
            if tag == "-":
                if block is None:
                    block = Block(old_ln)
                    cur.blocks.append(block)
                block.old.append(line[1:])
                old_ln += 1
                old_left -= 1
                continue
            if tag == "+":
                if block is None:
                    block = Block(old_ln)
                    cur.blocks.append(block)
                block.new.append(line[1:])
                new_left -= 1
                continue
            if tag == "\\":
                continue
            # context line (" " prefix; tolerate a stripped empty line)
            block = None
            old_ln += 1
            old_left -= 1
            new_left -= 1
            continue
        if line.startswith("diff --git "):
            m = _DIFF.match(line)
            cur = FileChange(m.group(1), m.group(2)) if m else FileChange(None, None)
            changes.append(cur)
            block = None
            continue
        if cur is None:
            continue
        if line.startswith("@@"):
            m = _HUNK.match(line)
            if not m:
                cur.status = "unknown"
                continue
            old_ln = int(m.group(1))
            old_left = int(m.group(2)) if m.group(2) is not None else 1
            new_left = int(m.group(4)) if m.group(4) is not None else 1
            if old_left == 0:
                old_ln += 1  # "-N,0" means the insertion goes after line N
            block = None
        elif line.startswith("new file mode"):
            cur.status = "new"
        elif line.startswith("deleted file mode"):
            cur.status = "delete"
        elif line.startswith("rename from") or line.startswith("rename to") or line.startswith("copy "):
            cur.status = "rename"
        elif line.startswith("Binary files") or line.startswith("GIT binary patch"):
            cur.status = "binary"
        elif line.startswith("--- "):
            p = line[4:].strip()
            if p != "/dev/null" and p.startswith("a/"):
                cur.old_path = p[2:]
        elif line.startswith("+++ "):
            p = line[4:].strip()
            if p == "/dev/null":
                cur.status = "delete"
            elif p.startswith("b/"):
                cur.path = p[2:]
    return changes


# ----------------------------------------------------------------------------- line helpers
def strip_comment(s):
    """The code part of one physical line (a '#' inside a string literal is not a comment)."""
    if "#" not in s:
        return s
    i, n, quote = 0, len(s), None
    while i < n:
        c = s[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if s.startswith(quote, i):
                i += len(quote)
                quote = None
                continue
            i += 1
            continue
        if c == "#":
            return s[:i]
        if c == '"' or c == "'":
            quote = s[i : i + 3] if s[i : i + 3] in ('"""', "'''") else c
            i += len(quote)
            continue
        i += 1
    return s


def code_lines(lines):
    out = []
    for text in lines:
        code = strip_comment(text).rstrip()
        if code.strip():
            out.append(code)
    return out


def _word(c):
    return c.isalnum() or c == "_"


def changed_columns(old, new):
    """Old-line column range [a, b) that differs from the new line, widened to whole words,
    and the old and new text of that range."""
    n = min(len(old), len(new))
    p = 0
    while p < n and old[p] == new[p]:
        p += 1
    s = 0
    while s < n - p and old[len(old) - 1 - s] == new[len(new) - 1 - s]:
        s += 1
    while p > 0 and (_word(old[p - 1]) or _word(new[p - 1])):
        p -= 1
    while s > 0 and (_word(old[len(old) - s]) or _word(new[len(new) - s])):
        s -= 1
    return p, len(old) - s, old[p : len(old) - s], new[p : len(new) - s]


class _Span:
    """A stand-in block covering old lines start .. start+count-1."""

    __slots__ = ("start", "old", "new")

    def __init__(self, start, count):
        self.start = start
        self.old = [None] * count
        self.new = []


def _apply(lines, blocks, first, last):
    """Lines first..last of the pinned file with the given blocks applied."""
    out = []
    pos = first
    for b in sorted(blocks, key=lambda b: b.start):
        out.extend(lines[pos - 1 : b.start - 1])
        out.extend(b.new)
        pos = b.start + len(b.old)
    out.extend(lines[pos - 1 : last])
    return out


def _dedent(lines):
    """Strips the first line's indentation from every line; None if a code line sits further
    left than that (the edit moved code out of its block)."""
    if not lines:
        return lines
    first = lines[0]
    indent = first[: len(first) - len(first.lstrip())]
    out = []
    for line in lines:
        if line.startswith(indent):
            out.append(line[len(indent):])
        elif not line.strip() or line.lstrip().startswith("#"):
            out.append(line.lstrip())
        else:
            return None
    return out


def _compiles(lines, path):
    if lines is None:
        return False
    try:
        compile("\n".join(lines), path, "exec", dont_inherit=True)
    except (SyntaxError, ValueError):
        return False
    return True


def _overlay_finder(overlay, root):
    """A meta path finder serving patched copies of some modules from ``overlay``."""
    import importlib.util

    patched = {}
    for base, _dirs, names in os.walk(overlay):
        for name in names:
            full = os.path.join(base, name)
            rel = os.path.relpath(full, overlay).replace(os.sep, "/")
            mod = _module_of(rel)
            locations = [os.path.join(root, os.path.dirname(rel))] if rel.endswith("__init__.py") else None
            patched[mod] = (full, locations)

    class Overlay:
        @staticmethod
        def find_spec(name, path=None, target=None):
            hit = patched.get(name)
            if hit is None:
                return None
            return importlib.util.spec_from_file_location(name, hit[0], submodule_search_locations=hit[1])

    return Overlay


def _module_of(path):
    name = path[:-3].replace("/", ".")
    return name[: -len(".__init__")] if name.endswith(".__init__") else name


# ----------------------------------------------------------------------------- selection
class Selector:
    def __init__(self, blob, sources):
        self.files = blob["files"]
        self.reads = blob["reads"]
        self.scopes = blob["scopes"]
        self.sets = blob["sets"]
        self.fresh = blob.get("fresh", [])
        self.per_query = blob.get("per_query")
        self.pq_bit = (1 << self.per_query["test"]) if self.per_query else 0
        self.core = set(blob.get("core", ())) or None
        self._warm = False
        self._baseline = None
        self.sources = sources
        self._lines = {}

    # Accumulates either a bitset over map test indices or the GLOBAL marker.
    def run(self, patch_text):
        # [tests, everything, queries of the per-query test, per-query test charged as a whole]
        acc = [0, False, set(), False]
        self.import_check = set()
        self.patched = {}
        self.changed = {}
        for change in parse_patch(patch_text):
            self.file_change(change, acc)
            if acc[1]:
                break
        if self.import_check and not acc[1]:
            if not self.imports_cleanly():
                log("patched modules no longer import:", sorted(self.import_check))
                acc[1] = True
            if self.core is None or any(_module_of(p) in self.core for p in self.import_check):
                for index in self.fresh:  # these import sqlglot in a fresh interpreter
                    acc[0] |= 1 << index
        if not acc[1] and acc[0] & self.pq_bit and not acc[3] and acc[2] and self.recheck_queries(acc[2]):
            acc[0] &= ~self.pq_bit
        return acc

    def recheck_queries(self, queries):
        """True if the per-query test still passes for every query the edit can reach: each
        one, run over the patched sources, gives the same result table and the same duckdb SQL
        as at the pinned commit. Only tried when that is much cheaper than running the test."""
        ref = self.per_query
        if any(k >= len(ref["rows"]) for k in queries):
            return False
        if any(path.startswith("tests/") for path in self.changed):
            return False  # the re-check replays the test as written at the pinned commit
        estimate = sum(ref["seconds"][k] for k in queries)
        if any(ref["rows"][k] is not None for k in queries):
            estimate += ref["load"]
        if estimate > PER_QUERY_BUDGET:
            log("per-query test: %d queries, about %.1f s to re-check; running it whole" % (len(queries), estimate))
            return False
        if not os.path.isdir(os.path.join(self.sources, "tests", "fixtures", "optimizer", "tpc-ds")):
            return False
        expected = {k: (ref["rows"][k], ref["duckdb"][k]) for k in queries}
        script = os.path.join(HERE, "tpcds_check.py")

        def same():
            import importlib.util

            spec = importlib.util.spec_from_file_location("tpcds_check", script)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            got = module.fingerprints(self.sources, list(queries))
            differ = sorted(k for k in expected if got.get(k) != expected[k])
            if differ:
                log("per-query test: queries", differ, "change with this patch")
            return not differ

        ok = self.in_patched_child(same)
        if ok:
            log("per-query test passes for queries", sorted(queries), "- left out")
        return ok

    def imports_cleanly(self):
        """An edit to an import-time value is charged to the tests that read it, which is only
        right if every module still imports. The patched modules and their importers are
        imported again over the pinned sources; a failure is confirmed in a fresh interpreter
        before it is believed."""
        if not os.path.isdir(self.sources):
            return True  # nothing to check against; keep the map's answer
        for path, (info, blocks) in self.changed.items():
            if self.source_lines(path, info) is None:
                return True
            self.patched_text(path, info, blocks)
        wanted = set()
        for path in self.import_check:
            wanted.add(_module_of(path))
            wanted.update(m for m in self.files[path].get("importers", ()) if not m.endswith("__main__"))
        import shutil
        import tempfile

        overlay = tempfile.mkdtemp(prefix="blast-")
        try:
            for path, text in self.patched.items():
                dest = os.path.join(overlay, path)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(dest, "w", encoding="utf-8") as f:
                    f.write(text)
            try:
                if hasattr(os, "fork") and self.forked_check(overlay, wanted):
                    return True
            except Exception as error:  # noqa: BLE001
                log("forked import check unavailable, using a fresh interpreter:", repr(error))
            return self.fresh_check(overlay, wanted)
        except Exception as error:  # noqa: BLE001
            log("import check failed to run:", repr(error))
            return False
        finally:
            shutil.rmtree(overlay, ignore_errors=True)

    def in_patched_child(self, check):
        """Runs ``check()`` in a forked child with the patched modules (and their importers)
        imported again over the pinned sources. False on any error or without fork."""
        if not hasattr(os, "fork") or not os.path.isdir(self.sources):
            return False
        for path, (info, blocks) in self.changed.items():
            if self.source_lines(path, info) is None:
                return False
            self.patched_text(path, info, blocks)
        wanted = set()
        for path in self.patched:
            wanted.add(_module_of(path))
            wanted.update(m for m in self.files[path].get("importers", ()) if not m.endswith("__main__"))
        import shutil
        import tempfile

        overlay = tempfile.mkdtemp(prefix="blast-")
        try:
            for path, text in self.patched.items():
                dest = os.path.join(overlay, path)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(dest, "w", encoding="utf-8") as f:
                    f.write(text)
            return self.forked_check(overlay, wanted, check)
        except Exception as error:  # noqa: BLE001
            log("patched child failed to run:", repr(error))
            return False
        finally:
            shutil.rmtree(overlay, ignore_errors=True)

    def forked_check(self, overlay, wanted, check=None):
        """Imports sqlglot once in this process, then re-imports the patched modules (and the
        modules that import them) in a forked child, which costs far less than a fresh start."""
        if not self._warm:
            self._warm = True
            sys.path.insert(0, self.sources)
            stdout, sys.stdout = sys.stdout, sys.stderr  # stdout carries the answers only
            try:
                import sqlglot  # noqa: F401
            except Exception:  # noqa: BLE001
                return False
            finally:
                sys.stdout = stdout
        sys.stdout.flush()
        sys.stderr.flush()
        pid = os.fork()
        if pid == 0:
            status = 1
            try:
                os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
                sys.stdout = sys.stderr
                import importlib

                scope = {_module_of(p) for p in self.patched} | wanted
                # a package imported again must not keep stale submodules around (they would be
                # missing as attributes of the new package object), so those go too
                for name in list(sys.modules):
                    if name in scope or any(name.startswith(pkg + ".") for pkg in scope):
                        del sys.modules[name]
                sys.meta_path.insert(0, _overlay_finder(overlay, self.sources))
                for name in ["sqlglot"] + sorted(wanted):
                    if name not in self.baseline():
                        importlib.import_module(name)
                status = 0 if check is None or check() else 1
            except BaseException as error:  # noqa: BLE001
                if check is not None:
                    log("patched child:", repr(error)[:300])
                status = 1
            finally:
                os._exit(status)
        _, code = os.waitpid(pid, 0)
        return os.WIFEXITED(code) and os.WEXITSTATUS(code) == 0

    def fresh_check(self, overlay, wanted):
        import subprocess

        done = subprocess.run(
            [
                sys.executable, "-S", "-B", "-c", IMPORT_CHECK, self.sources, overlay,
                ",".join(self.baseline()), ",".join(sorted(wanted)),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return done.returncode == 0 and not done.stdout.strip()

    def baseline(self):
        """Modules that already fail to import in this image without any patch."""
        if self._baseline is None:
            try:
                with open(IMPORT_BASELINE) as f:
                    self._baseline = [m for m in f.read().split(",") if m.strip()]
            except OSError:
                self._baseline = []
        return self._baseline

    def charge(self, acc, sid, queries=None):
        """Adds a test set. ``queries`` says which queries of the per-query test the edit can
        reach through this set; without it, that test is charged as a whole."""
        if sid == GLOBAL:
            acc[1] = True
        elif sid >= 0:
            bits = self.sets[sid]
            acc[0] |= bits
            if self.pq_bit and bits & self.pq_bit:
                if queries is None:
                    acc[3] = True
                else:
                    acc[2].update(queries)

    def file_change(self, ch, acc):
        path = ch.path or ch.old_path
        if path is None or ch.status == "unknown":
            acc[1] = True
            return
        base = path.rsplit("/", 1)[-1]
        if base in CONFIG_FILES:
            acc[1] = True
            return
        if path.endswith(".py"):
            info = self.files.get(path)
            if ch.status in ("delete", "rename", "binary") or (info is None and ch.status != "new"):
                self.module_level(path, info, acc)
                return
            if info is None:  # a new module nothing imports yet
                if path.startswith("sqlglot/"):
                    acc[1] = True
                return
            self.python_change(path, info, ch.blocks, acc)
            return
        self.data_change(path, ch, acc)

    def data_change(self, path, ch, acc):
        self.charge(acc, self.reads.get(path, NO_TESTS))
        if ch.old_path and ch.old_path != path:
            self.charge(acc, self.reads.get(ch.old_path, NO_TESTS))
        if ch.status in ("new", "delete", "rename"):
            parts = path.split("/")
            for i in range(1, len(parts)):
                self.charge(acc, self.reads.get("/".join(parts[:i]) + "/", NO_TESTS))

    def module_level(self, path, info, acc):
        """An edit that runs when the module is imported."""
        if path.startswith("tests/"):
            sid = self.scopes.get(path)
            if sid is None:
                acc[1] = True
            else:
                self.charge(acc, sid)
        elif path.startswith("sqlglot/") or info is None:
            acc[1] = True
        else:
            self.charge(acc, info["users"])

    def source_lines(self, path, info):
        if path not in self._lines:
            lines = None
            src = os.path.join(self.sources, path)
            try:
                with open(src, "rb") as f:
                    raw = f.read()
                if self.sha1(raw) == info["sha1"]:
                    lines = raw.decode("utf-8").split("\n")
            except OSError:
                pass
            self._lines[path] = lines
        return self._lines[path]

    @staticmethod
    def sha1(raw):
        import hashlib

        return hashlib.sha1(raw).hexdigest()

    def inert(self, info, b):
        """True if the block only adds, removes or rewrites comments and blank lines."""
        flags = info["flags"]
        n = len(flags)
        for i in range(len(b.old)):
            ln = b.start + i
            if 1 <= ln <= n and flags[ln - 1] & (IN_STRING | CONTINUED):
                return False
        anchor = b.start - 1
        if 1 <= anchor <= n and flags[anchor - 1] & (STRING_AFTER | CONTINUED):
            return False
        if b.start <= 2 and any(CODING_RE.match(t) for t in b.old + b.new):
            return False
        return code_lines(b.old) == code_lines(b.new)

    def pure_duplicate(self, path, info, b):
        """True if the block repeats whole class or module level assignments of plain values
        (``X = True`` twice in a row), which leaves every value as it was."""
        lines = self.source_lines(path, info)
        if lines is None:
            return False
        if b.old and b.new == b.old + b.old:
            first, count = b.start, len(b.old)
        elif not b.old and b.new and lines[b.start - 1 - len(b.new) : b.start - 1] == b.new:
            first, count = b.start - len(b.new), len(b.new)
        elif not b.old and b.new and lines[b.start - 1 : b.start - 1 + len(b.new)] == b.new:
            first, count = b.start, len(b.new)
        else:
            return False
        units = self.units_for(info, [_Span(first, count)])
        if units is None or list(units) != [(first, first + count - 1)]:
            return False  # not exactly one complete statement
        pure = info.get("pure") or ()
        return all(ln in pure and ln in info["cattr"] for ln in range(first, first + count))

    def dropped_statements(self, info, b):
        """Set ids of the class or module level assignments a pure deletion removes, if it
        removes whole assignment statements and nothing else; otherwise None."""
        units = self.units_for(info, [b])
        if units is None:
            return None
        sids = []
        for (u0, u1), _inner in units.items():
            if not (b.start <= u0 and u1 <= b.start + len(b.old) - 1):
                return None
        for ln in range(b.start, b.start + len(b.old)):
            entry = info["cattr"].get(ln)
            if entry is None or not ASSIGN_TARGET_RE.match(b.old[0]):
                return None
            sids.append(entry[0])
        return sids

    @staticmethod
    def whole_entry_edit(info, b):
        """True if a block only adds or removes whole lines strictly inside one class or module
        level assignment (entries of a table, not its first line), so the change is a change to
        that one attribute's value."""
        cattr = info["cattr"]
        if b.old:
            lines = range(b.start, b.start + len(b.old))
            neighbours = (b.start - 1, b.start + len(b.old))
        else:
            lines = ()
            neighbours = (b.start - 1, b.start)
        owners = set()
        for ln in list(lines) + list(neighbours):
            entry = cattr.get(ln)
            if entry is None or entry[0] == GLOBAL:
                return False
            owners.add(entry[0])
        if len(owners) != 1:
            return False
        texts = list(b.old) + list(b.new)
        return not any(ASSIGN_TARGET_RE.match(t) for t in texts)

    @staticmethod
    def weak_match(sites, old, new):
        """If the edit swaps exactly one profiled operator or constant, the tests that would
        have seen that swap change a result (from the weak mutation profile)."""
        for c0, c1, text, variants in sites:
            if old[c0:c1] != text:
                continue
            for vtext, sid, *queries in variants:
                q = queries[0] if queries else None
                if old[:c0] + vtext + old[c1:] == new:
                    return sid, q
                if vtext in ("==", "!=") and old[:c0].rstrip() + " " + vtext + " " + old[c1:].lstrip() == new:
                    return sid, q
        return None

    def compiles(self, path, info, blocks):
        """Syntax check of the patched file; an edit that breaks the syntax breaks every
        importer. Each edit is checked by compiling just the unit around it (a module-level
        statement or one statement of a class body); anything unusual compiles the whole file."""
        lines = self.source_lines(path, info)
        if lines is None:
            return True  # no pinned source to check against; keep the map's answer
        for b in blocks:
            if lines[b.start - 1 : b.start - 1 + len(b.old)] != b.old:
                return False  # the patch does not apply to the pinned file
        units = self.units_for(info, blocks)
        if units is not None:
            ok = True
            for (u0, u1), inner in units.items():
                if not _compiles(_dedent(_apply(lines, inner, u0, u1)), path):
                    ok = False
                    break
            if ok:
                return True
        return _compiles(self.patched_text(path, info, blocks).split("\n"), path)

    @staticmethod
    def units_for(info, blocks):
        """{(first, last line): blocks inside it} or None if some edit is not inside one unit."""
        import bisect

        flat = info.get("units") or []
        starts = flat[0::2]
        ends = flat[1::2]
        out = {}
        for b in blocks:
            lo = b.start if b.old else b.start - 1
            hi = b.start + len(b.old) - 1 if b.old else b.start
            i = bisect.bisect_right(starts, lo) - 1
            if i < 0 or not (starts[i] <= lo and hi <= ends[i]):
                return None
            out.setdefault((starts[i], ends[i]), []).append(b)
        return out

    def patched_text(self, path, info, blocks):
        text = self.patched.get(path)
        if text is None:
            lines = self.source_lines(path, info)
            text = "\n".join(_apply(lines, blocks, 1, len(lines)))
            self.patched[path] = text
        return text

    def python_change(self, path, info, blocks, acc):
        kind = info["kind"]
        n = len(kind)
        self.changed[path] = (info, blocks)
        live = [b for b in blocks if not self.inert(info, b) and not self.pure_duplicate(path, info, b)]
        if not live:
            return
        if not self.compiles(path, info, blocks):
            log(path, "does not compile after the patch")
            self.module_level(path, info, acc)
            return
        sets = info["sets"]
        for b in live:
            pairs = len(b.old) == len(b.new)
            if b.old:
                lines = range(b.start, b.start + len(b.old))
            else:
                lines = (b.start - 1, b.start)
            for i, ln in enumerate(lines):
                if not 1 <= ln <= n:
                    continue
                k = kind[ln - 1]
                if pairs and ln in info.get("annot", ()):
                    a0, a1, _o, _n = changed_columns(b.old[i], b.new[i])
                    if any(c0 <= a0 and max(a1, a0 + 1) <= c1 for c0, c1 in info["annot"][ln]):
                        continue  # an annotation nothing evaluates
                if k in (RUNTIME, MIXED) and pairs and len(b.old) == 1 and ln in info["weak"]:
                    hit = self.weak_match(info["weak"][ln], b.old[0], b.new[0])
                    if hit is not None:
                        self.charge(acc, hit[0], hit[1])
                        continue
                qsets = info.get("qsets", {})
                if k == RUNTIME:
                    self.charge(acc, sets.get(ln, NO_TESTS), qsets.get(ln))
                    if ln in info["checks"]:
                        self.import_check.add(path)
                    continue
                cols = None
                if pairs and b.old:
                    cols = changed_columns(b.old[i], b.new[i])
                if k == MIXED and cols is not None:
                    a, z = cols[0], max(cols[1], cols[0] + 1)
                    if any(c0 <= a and z <= c1 for c0, c1 in info["cols"].get(ln, ())):
                        self.charge(acc, sets.get(ln, NO_TESTS), qsets.get(ln))
                        if ln in info["checks"]:
                            self.import_check.add(path)
                        continue
                if k == DOC and cols is not None and cols[2].count('"') == cols[3].count('"') and cols[2].count("'") == cols[3].count("'"):
                    continue
                if k == IMPORT and cols is not None and ln in info["argt"]:
                    if SIMPLE_VALUE_RE.fullmatch(cols[2].strip()) and SIMPLE_VALUE_RE.fullmatch(cols[3].strip()):
                        self.charge(acc, info["argt"][ln])
                        continue
                if k in (IMPORT, MIXED) and cols is not None and ln in info["dflt"]:
                    a, z = cols[0], max(cols[1], cols[0] + 1)
                    hit = [(sid, check) for c0, c1, sid, check in info["dflt"][ln] if c0 <= a and z <= c1]
                    if hit:
                        for sid, check in hit:
                            self.charge(acc, sid)
                            if check:
                                self.import_check.add(path)
                        continue
                if k in (IMPORT, MIXED) and cols is not None and ln in info["cattr"]:
                    target = ASSIGN_TARGET_RE.match(b.old[i])
                    sid, check = info["cattr"][ln]
                    if target is None or cols[0] >= target.end():
                        a, z = cols[0], max(cols[1], cols[0] + 1)
                        entry = [e for e in info.get("kattr", {}).get(ln, ()) if e[0] <= a and z <= e[1]]
                        if entry:
                            sid = entry[0][2]  # the value of one dict entry: whoever looked up its key
                        self.charge(acc, sid)  # the value changed
                        if check:
                            self.import_check.add(path)
                        continue
                    renamed = ASSIGN_TARGET_RE.match(b.new[i])
                    if renamed is None or renamed.group(1) == target.group(1):
                        # the assignment was dropped or rewritten in place: same attribute
                        self.charge(acc, sid)
                        self.import_check.add(path)
                        continue
                if not b.old and k in (NONE, DOC) and not code_lines(b.new):
                    continue
                if not b.new and k in (IMPORT, MIXED):
                    dropped = self.dropped_statements(info, b)
                    if dropped is not None:
                        for sid in dropped:
                            self.charge(acc, sid)
                        self.import_check.add(path)
                        continue
                if not pairs and k in (IMPORT, MIXED) and self.whole_entry_edit(info, b):
                    # whole lines added to or removed from the middle of a class-level table
                    sid, _check = info["cattr"][ln]
                    self.charge(acc, sid)
                    self.import_check.add(path)
                    continue
                self.module_level(path, info, acc)
            if acc[1]:
                return


# ----------------------------------------------------------------------------- packing
def split(numbers, ms):
    """Spread tests over 1..8 workers, minimising the busiest worker plus 2 s per worker."""
    if not numbers:
        return []
    items = sorted(numbers, key=lambda t: (-ms[t], t))
    total = sum(ms[t] for t in items)
    best = None
    for k in range(1, min(MAX_WORKERS, len(items)) + 1):
        bound = max(ms[items[0]], -(-total // k)) + WORKER_MS * k
        if best is not None and bound >= best[0]:
            continue
        heap = [(0, i) for i in range(k)]
        bins = [[] for _ in range(k)]
        loads = [0] * k
        for t in items:
            load, i = heapq.heappop(heap)
            bins[i].append(t)
            loads[i] = load + ms[t]
            heapq.heappush(heap, (loads[i], i))
        if len(items) <= 400:
            _improve(bins, loads, ms)
        used = [b for b in bins if b]
        cost = max(loads) + WORKER_MS * len(used)
        if best is None or cost < best[0]:
            best = (cost, used)
    return best[1]


def _improve(bins, loads, ms, rounds=64):
    """Moves and swaps out of the busiest worker while they lower its load."""
    k = len(bins)
    if k < 2:
        return
    for _ in range(rounds):
        hi = max(range(k), key=loads.__getitem__)
        top = loads[hi]
        done = False
        for lo in sorted(range(k), key=loads.__getitem__):
            if lo == hi:
                continue
            gap = top - loads[lo]
            for t in bins[hi]:
                d = ms[t]
                if 0 < d < gap:
                    bins[hi].remove(t)
                    bins[lo].append(t)
                    loads[hi] -= d
                    loads[lo] += d
                    done = True
                    break
            if done:
                break
            for t in bins[hi]:
                for u in bins[lo]:
                    delta = ms[t] - ms[u]
                    if 0 < delta < gap:
                        bins[hi].remove(t)
                        bins[lo].remove(u)
                        bins[hi].append(u)
                        bins[lo].append(t)
                        loads[hi] -= delta
                        loads[lo] += delta
                        done = True
                        break
                if done:
                    break
            if done:
                break
        if not done:
            return


# ----------------------------------------------------------------------------- main
def main():
    with open(os.path.join(DATA, "tests.json"), "rb") as f:
        listing = json.load(f)
    tests = listing["tests"]
    ms = {t["number"]: int(round(t["seconds"] * 1000)) for t in tests}
    every = sorted(ms)

    selector = None
    index_to_number = []
    unknown = []
    try:
        with open(MAP_PATH, "rb") as f:
            blob = marshal.loads(zlib.decompress(f.read()))
        number_of = {t["id"]: t["number"] for t in tests}
        index_to_number = [number_of.get(t, -1) for t in blob["tests"]]
        mapped = set(index_to_number)
        unknown = [t for t in every if t not in mapped]
        if unknown:
            log(len(unknown), "tests are not in the map; they run on every commit")
        selector = Selector(blob, SOURCE_ROOT)
    except Exception as error:  # noqa: BLE001
        log("map unavailable, running everything:", repr(error))

    commits = sorted(name[:-6] for name in os.listdir(os.path.join(DATA, "commits")) if name.endswith(".patch"))
    cache = {}
    out = []
    for cid in commits:
        try:
            with open(os.path.join(DATA, "commits", cid + ".patch"), encoding="utf-8", errors="replace") as f:
                text = f.read()
            if selector is None:
                raise RuntimeError("no map")
            bits, everything = selector.run(text)[:2]
        except Exception as error:  # noqa: BLE001
            log(cid, "selection failed, running everything:", repr(error))
            bits, everything = 0, True
        key = None if everything else bits
        workers = cache.get(key)
        if workers is None:
            if everything:
                chosen = every
            else:
                chosen = set(unknown)
                while bits:
                    low = bits & -bits
                    number = index_to_number[low.bit_length() - 1]
                    if number >= 0:
                        chosen.add(number)
                    bits ^= low
            workers = cache[key] = split(chosen, ms)
        out.append(json.dumps({"commit": cid, "workers": workers}, separators=(",", ":")))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()
