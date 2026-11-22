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

import ast
import heapq
import json
import marshal
import os
import re
import sys
import types
import unittest  # noqa: F401  sqlglot turns on its strict argument checks when tests run; so do we
import zlib

import state_diff
import tpcds_check

DATA = os.environ.get("BLAST_DATA", "/data")
HERE = os.path.dirname(os.path.abspath(__file__))
MAP_PATH = os.environ.get("BLAST_MAP", os.path.join(HERE, "map.bin"))
SOURCE_ROOT = os.environ.get("BLAST_SOURCE", os.path.join(HERE, "sqlglot-src"))

NO_TESTS, GLOBAL = -1, -2
NONE, RUNTIME, IMPORT, MIXED, DOC = 0, 1, 2, 3, 4
IN_STRING, STRING_AFTER, CONTINUED = 1, 2, 4
WORKER_MS = 2000
MAX_WORKERS = 8
WHOLE_RECHECK_SAVING = 10.0  # a re-check of every query, only for at least this many seconds
WHOLE_RECHECK_SECONDS = 30.0  # the child doing it
WHOLE_RECHECK_BUDGET = 40.0  # its estimate (build-time seconds, about twice the actual cost)
PER_QUERY_BUDGET = 16.0  # most seconds of re-checking ever worth trying instead of a 21 s test
CHILD_SECONDS = 20.0  # longest any single patched child may run before it is killed
IMPORT_SECONDS = 8.0  # an import check normally takes well under a second
TOTAL_CHILD_SECONDS = 100.0  # all patched children together, for the whole batch
STATE_DIFF_MODULES = 250  # fingerprint at most this many modules to measure an import-time edit
STATE_DIFF_SECONDS = 20.0

CONFIG_FILES = {"pyproject.toml", "setup.cfg", "tox.ini", "pytest.ini", "conftest.py", ".coveragerc"}
CODING_RE = re.compile(r"^[ \t\f]*#.*?coding[:=]")
SIMPLE_VALUE_RE = re.compile(r"True|False|None|[-+]?\d+")
# A replacement value that cannot raise or name anything when evaluated.
LITERAL_RE = re.compile(r"""\s*(True|False|None|[-+]?\d+(\.\d+)?|"[^"\\]*"|'[^'\\]*')?\s*""")
ASSIGN_TARGET_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?::[^=]*)?=(?!=)")
IMPORT_BASELINE = os.environ.get("BLAST_IMPORT_BASELINE", os.path.join(HERE, "import_baseline.txt"))
# Digests of each TPC-DS query's plan, computed in the image at build time (tpcds_check.py --plans),
# and the checkpoints of the steps that lead to it.
PLANS_PATH = os.environ.get("BLAST_TPCDS_PLANS", os.path.join(HERE, "tpcds_plans.json"))
POINTS_PATH = os.path.splitext(PLANS_PATH)[0] + "_points.pickle"
# What test_max_nodes sees on scaled copies of its input at the pinned commit (max_nodes_check.py).
MAX_NODES_REF = os.environ.get("BLAST_MAX_NODES_REF", os.path.join(HERE, "max_nodes_ref.json"))
MAX_NODES_TEST = "tests/test_parser.py::TestParser::test_max_nodes"
# A selection this small is cheaper to run here than to pay a worker for: dialect tests only
# (they need nothing beyond sqlglot and the standard library), at most this many, recorded at
# most this long together.
RERUN_MAX_TESTS = 8
RERUN_MAX_MS = 1000
RERUN_DIRS = ("tests/dialects/",)
RERUN_UNAVAILABLE = ("tests/dialects/test_bigquery.py",)  # needs pytz

# Imports the given sqlglot modules with some files replaced by patched copies, and prints the
# ones that fail. argv: <pinned sources> <overlay dir> <modules to skip> <modules to import>
IMPORT_CHECK = r"""
import importlib, importlib.util, os, sys, unittest
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
for name in ["sqlglot"] + wanted:
    if name in skip:
        continue
    try:
        importlib.import_module(name)
    except BaseException:
        print(name)  # one failure settles it
        break
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
    __slots__ = ("path", "old_path", "status", "blocks", "hunked")

    def __init__(self, old_path, path):
        self.old_path = old_path
        self.path = path
        self.status = "modify"
        self.blocks = []
        self.hunked = False


_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_DIFF = re.compile(r"^diff --git a/(.*) b/(.*)$")


def _header_path(line, prefix):
    """The path of a ---/+++ header line, without its a/ or b/ prefix or a trailing timestamp."""
    p = line[4:].split("\t", 1)[0].strip()
    if p == "/dev/null":
        return p
    return p[len(prefix):] if p.startswith(prefix) else p


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
        if line.startswith("--- ") and (cur is None or cur.hunked):
            # a plain unified diff: no "diff --git" line, files start at their "---" header
            cur = FileChange(None, None)
            changes.append(cur)
            block = None
        if cur is None:
            continue
        if line.startswith("@@"):
            m = _HUNK.match(line)
            if not m:
                cur.status = "unknown"
                continue
            cur.hunked = True
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
            p = _header_path(line, "a/")
            if p == "/dev/null":
                cur.status = "new"
            elif p:
                cur.old_path = p
                cur.path = cur.path or p
        elif line.startswith("+++ "):
            p = _header_path(line, "b/")
            if p == "/dev/null":
                cur.status = "delete"
            elif p:
                cur.path = p
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


def _own_state(stmt, bound):
    """A top-level statement whose only effect is on its module's own namespace."""
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        # decorators run arbitrary code on the definition (registries); only plain names that
        # sqlglot uses to annotate are trusted
        return all(isinstance(d, (ast.Name, ast.Attribute)) and ast.unparse(d) in TRUSTED_DECORATORS for d in stmt.decorator_list)
    if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.Pass)):
        return True
    if isinstance(stmt, ast.Expr):
        return isinstance(stmt.value, ast.Constant)  # a docstring
    if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        for t in targets:
            for sub in ast.walk(t):
                if isinstance(sub, (ast.Attribute, ast.Subscript)):
                    base = sub.value
                    while isinstance(base, (ast.Attribute, ast.Subscript)):
                        base = base.value
                    if not (isinstance(base, ast.Name) and base.id in bound):
                        return False  # stores into another module's object
        return True
    if isinstance(stmt, (ast.If, ast.Try)):
        parts = stmt.body + stmt.orelse + getattr(stmt, "finalbody", [])
        for handler in getattr(stmt, "handlers", []):
            parts += handler.body
        return all(_own_state(x, bound) for x in parts)
    return False


TRUSTED_DECORATORS = {"staticmethod", "classmethod", "property", "t.overload", "typing.overload", "overload",
                      "dataclass", "dataclasses.dataclass", "functools.cache", "functools.lru_cache", "lru_cache", "cache"}


def _defs(tree):
    """Every def and class of a module by qualified name ("C.f", "f.<locals>.g")."""
    out = {}

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + child.name
                if name in out:
                    out[name + "#%d" % len(out)] = child  # a redefinition: keep both, in order
                else:
                    out[name] = child
                inner = name + (".<locals>." if not isinstance(child, ast.ClassDef) else ".")
                walk(child, inner)
            elif isinstance(child, ast.stmt) or isinstance(child, ast.excepthandler) or isinstance(child, ast.match_case):
                walk(child, prefix)

    walk(tree, "")
    return out


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


def _bindings(lines):
    """For every function and comprehension scope in a unit, the names it binds locally,
    its parameters, and those it declares global or nonlocal."""
    import symtable

    out = []

    def walk(table, path):
        if table.get_type() != "module":
            names = sorted(
                (sym.get_name(), sym.is_local(), sym.is_parameter(), sym.is_declared_global(), sym.is_nonlocal())
                for sym in table.get_symbols()
                if sym.is_local() or sym.is_parameter() or sym.is_declared_global() or sym.is_nonlocal()
            )
            out.append((path, table.get_name(), names))
        for child in table.get_children():
            walk(child, path + (table.get_name(),))

    walk(symtable.symtable("\n".join(lines), "<unit>", "exec"), ())
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
        self.class_data = blob.get("class_data", {})
        self.attr_readers = blob.get("attr_readers", {})
        self.key_readers = blob.get("key_readers", {})
        self.global_readers = blob.get("global_readers")
        self.global_external = blob.get("global_external", {})
        self.attr_queries = blob.get("attr_queries", {})
        self._pristine = {}
        self.source_readers = blob.get("source_readers", {})
        self.test_ids = list(blob.get("tests", ()))
        self.mx_bit = 0
        self.mx_ref = None
        if MAX_NODES_TEST in blob.get("tests", ()):
            try:
                with open(MAX_NODES_REF) as f:
                    self.mx_ref = json.load(f)
                self.mx_bit = 1 << list(blob["tests"]).index(MAX_NODES_TEST)
            except (OSError, ValueError):
                pass
        self.scopes = blob["scopes"]
        self.sets = blob["sets"]
        self.fresh = blob.get("fresh", [])
        self.per_query = blob.get("per_query")
        self.pq_bit = (1 << self.per_query["test"]) if self.per_query else 0
        self.plans = None
        if self.per_query and self.per_query.get("staged"):
            try:
                with open(PLANS_PATH) as f:
                    plans = json.load(f)
                if len(plans["plans"]) == len(self.per_query["rows"]) == len(plans["seconds"]):
                    self.plans = plans
            except (OSError, ValueError, KeyError, TypeError):
                pass
        self.core = set(blob.get("core", ())) or None
        self._warm = False
        self._baseline = None
        self.child_seconds = 0.0
        self.ms = {}
        self.index_to_number = []
        self.sources = sources
        self._lines = {}

    # Accumulates either a bitset over map test indices or the GLOBAL marker.
    def run(self, patch_text):
        # [tests, everything, queries of the per-query test, per-query test charged as a whole,
        #  those queries the edit reaches after their plan exists, steps it reaches before that]
        acc = [0, False, set(), False, set(), 0]
        self.why = None
        self.import_check = set()
        self.touched = set()
        self.unmodeled = set()
        self.import_hung = False
        self.import_values = set()
        self.patched = {}
        self.changed = {}
        for change in parse_patch(patch_text):
            self.file_change(change, acc)
            if acc[1]:
                break
        if self.import_check and not acc[1]:
            if not self.imports_cleanly():
                log("patched modules no longer import:", sorted(self.import_check))
                if self.import_hung:
                    self._mark(acc, "check")
                self.why = self.why or ("check" if acc[1] else None)
                for path in sorted(self.import_check):
                    self._checking = True
                    self.unimportable(path, self.files.get(path), acc)
                    self._checking = False
                if acc[1]:
                    self.why = self.why or "check"
            elif self.unmodeled and not self.measured_change(acc):
                log("import-time edits nothing narrower covers, in", sorted(self.unmodeled), "- running everything")
                self._mark(acc, "unmodeled")
            if self.core is None or any(_module_of(p) in self.core for p in self.import_check):
                for index in self.fresh:  # these import sqlglot in a fresh interpreter
                    acc[0] |= 1 << index
        if not acc[1] and acc[0] and self.rerun_passes(acc[0]):
            acc[0] = 0
            return acc
        # test_max_nodes first: while it is in, leaving out the per-query test saves little
        if not acc[1] and acc[0] & self.mx_bit and self.max_nodes_recheck():
            acc[0] &= ~self.mx_bit
        if not acc[1] and acc[0] & self.pq_bit and not acc[3] and acc[2]:
            if self.recheck_queries(acc[2], self.saving_without(acc[0], self.pq_bit), acc[4], acc[5],
                                    sorted(self.import_values)):
                acc[0] &= ~self.pq_bit
        elif not acc[1] and acc[0] & self.pq_bit and acc[3] and self.per_query:
            # reached as a whole: every query is re-checked, when leaving the test out saves
            # enough (a full re-check costs well under the test's own 21 s)
            saving = self.saving_without(acc[0], self.pq_bit)
            # never at the expense of the import checks of the commits still to come
            spare = TOTAL_CHILD_SECONDS - self.child_seconds
            if saving >= WHOLE_RECHECK_SAVING and spare > 2 * WHOLE_RECHECK_SECONDS:
                every = range(len(self.per_query["rows"]))
                if self.recheck_queries(every, saving, None, 0, sorted(self.import_values), whole=True):
                    acc[0] &= ~self.pq_bit
                else:
                    log("per-query test reached as a whole: kept")
        return acc

    def saving_without(self, bits, drop):
        """Seconds the commit's schedule gets cheaper if the tests in ``drop`` are left out."""
        if not self.ms:
            return float("inf")

        def numbers(b):
            out = set()
            while b:
                low = b & -b
                n = self.index_to_number[low.bit_length() - 1]
                if n >= 0:
                    out.add(n)
                b ^= low
            return out

        return (_cost(split(numbers(bits), self.ms), self.ms) - _cost(split(numbers(bits & ~drop), self.ms), self.ms)) / 1000

    def recheck_queries(self, queries, saving=float("inf"), back=None, steps=0, values=(), whole=False):
        """True if the per-query test still passes for every query the edit can reach.

        A query the edit only reaches before its plan exists (parsing, optimizing, planning)
        passes if its plan and its duckdb SQL come out exactly as at the pinned commit: what
        runs after that never meets the edit and gets the same input. When the edit only reaches
        some optimizer rules or the planner (``steps``), those alone are replayed from
        checkpoints of their inputs, and must give exactly the outputs they gave. Any other
        query, and any whose plan did change, passes if the patched sources give the same result
        table and the same duckdb SQL. Only tried when that is much cheaper than running the test."""
        ref = self.per_query
        queries = set(queries)
        if any(k >= len(ref["rows"]) for k in queries):
            return False
        if any(path.startswith("tests/") for path in self.changed):
            return False  # the re-check replays the test as written at the pinned commit
        for p in self.touched:
            if p.startswith("sqlglot/") and p.endswith(".py"):
                continue
            # ... with the pinned fixtures and data too: not for a file the test reads
            sid = self.reads.get(p, NO_TESTS)
            if p.endswith(".py") or sid == GLOBAL or (sid >= 0 and self.sets[sid] & self.pq_bit):
                return False
            parts = p.split("/")
            for i in range(1, len(parts)):
                d = self.reads.get("/".join(parts[:i]) + "/", NO_TESTS)
                if d == GLOBAL or (d >= 0 and self.sets[d] & self.pq_bit):
                    return False
        if self.plans is None or back is None:
            full = set(queries)
        else:
            full = queries & set(back)
        cheap = queries - full
        plans = self.plans
        step_seconds = (plans or {}).get("step_seconds") or [None] * len(ref["rows"])
        replay = {}
        for k in cheap:
            cost = tpcds_check.replay_cost(step_seconds[k], steps) if ref["rows"][k] is not None else None
            replay[k] = cost

        def run_cost(ks, loaded=False):
            cost = sum(ref["seconds"][k] for k in ks)
            if not loaded and any(ref["rows"][k] is not None for k in ks):
                cost += ref["load"]
            return cost

        estimate = run_cost(full) + sum(plans["seconds"][k] if replay[k] is None else replay[k] for k in cheap)
        limit = WHOLE_RECHECK_BUDGET if whole else min(PER_QUERY_BUDGET, 0.6 * saving)
        if estimate > limit:
            log("per-query test: %d queries (%d before planning only), about %.1f s to re-check for %.1f s saved; "
                "running it whole" % (len(queries), len(cheap), estimate, saving))
            return False
        if not os.path.isdir(os.path.join(self.sources, "tests", "fixtures", "optimizer", "tpc-ds")):
            return False
        script = os.path.join(HERE, "tpcds_check.py")
        by_points = sorted((k for k in cheap if replay[k] is not None), key=lambda q: replay[q])

        def snapshot():
            # in the child, before the patched modules replace the pinned ones
            return {v: tpcds_check.value_digest(tpcds_check.import_value(v)) for v in values}

        def same(before=None):
            import importlib.util
            import pickle
            import signal

            spec = importlib.util.spec_from_file_location("tpcds_check", script)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if values:
                for v in values:
                    if module.value_digest(module.import_value(v)) != before.get(v):
                        log("per-query test: what the import computed with the edited code differs (%s.%s)"
                            % (v[0] + ("." + v[1] if v[1] else ""), v[2]))
                        return False

            class OverTime(Exception):
                pass

            def over_time(_signum, _frame):
                raise OverTime

            def within(seconds, call, *args):
                """call(*args), or OverTime once it has used ``seconds`` of CPU: a patched plan
                can take far longer to run than the original (a join turned into a cross join)."""
                signal.signal(signal.SIGPROF, over_time)
                signal.setitimer(signal.ITIMER_PROF, max(seconds, 0.05))
                try:
                    return call(*args)
                finally:
                    signal.setitimer(signal.ITIMER_PROF, 0)

            state = {"spent": estimate, "loaded": any(ref["rows"][k] is not None for k in full)}

            def results_same(k):
                # something before the plan changed: only the result can tell whether the test
                # still passes
                cost = run_cost([k], state["loaded"])
                state["spent"] += cost
                state["loaded"] = state["loaded"] or ref["rows"][k] is not None
                if state["spent"] > limit:
                    log("per-query test: query", k, "changes before planning; no time left to compare results")
                    return False
                try:
                    got = within(3 * cost + 0.5, module.fingerprints, self.sources, [k])[k]
                except OverTime:
                    log("per-query test: query", k, "changes before planning and runs far longer")
                    return False
                if got != (ref["rows"][k], ref["duckdb"][k]):
                    log("per-query test: query", k, "changes with this patch")
                    return False
                return True

            fronts = set(cheap) - set(by_points)
            if by_points:
                with open(POINTS_PATH, "rb") as f:
                    points = pickle.load(f)
                for k in by_points:
                    verdict = module.checkpoint_check(self.sources, points[k], steps) if k in points else None
                    if verdict is None:
                        fronts.add(k)  # the checkpoints cannot answer: a whole front pass can
                        state["spent"] += plans["seconds"][k]
                    elif not verdict and not results_same(k):
                        return False
            for k, got in module.iter_front_digests(self.sources, sorted(fronts, key=lambda q: plans["seconds"][q])):
                if got != (plans["plans"][k], plans["duckdb"][k]) and not results_same(k):
                    return False
            if full:
                try:
                    got = within(3 * run_cost(full) + 1.0, module.fingerprints, self.sources, sorted(full))
                except OverTime:
                    log("per-query test: queries", sorted(full), "run far longer with this patch")
                    return False
                differ = sorted(k for k in full if got.get(k) != (ref["rows"][k], ref["duckdb"][k]))
                if differ:
                    log("per-query test: queries", differ, "change with this patch")
                    return False
            return True

        cap = WHOLE_RECHECK_SECONDS if whole else min(CHILD_SECONDS, 3 * limit + 3)
        ok = self.in_patched_child(same, timeout=cap, snapshot=snapshot if values else None)
        if ok:
            log("per-query test passes for queries", sorted(queries),
                "(%d by checkpoints, %d by plan) - left out" % (len(by_points), len(cheap) - len(by_points)))
        return ok

    def rerun_passes(self, bits):
        """True if a small selection of dialect tests, run with unittest over the patched
        sources in a forked child, passes as a whole: every test runs, none fails, errors or is
        skipped, no subtest fails. Only for edits to sqlglot's Python files."""
        # the re-run patches Python files only: every touched file must be one of sqlglot's
        if not self.changed or not self.touched or any(
                not (p.startswith("sqlglot/") and p.endswith(".py")) for p in self.touched):
            return False
        ids, total = [], 0
        while bits:
            low = bits & -bits
            i = low.bit_length() - 1
            bits ^= low
            if i >= len(self.test_ids) or i >= len(self.index_to_number):
                return False
            test = self.test_ids[i]
            number = self.index_to_number[i]
            if number < 0 or not test.startswith(RERUN_DIRS) or test.split("::")[0] in RERUN_UNAVAILABLE:
                return False
            total += self.ms.get(number, 1 << 30)
            ids.append(test)
            if len(ids) > RERUN_MAX_TESTS or total > RERUN_MAX_MS:
                return False
        if not ids:
            return False
        names = []
        for test in ids:
            parts = test.split("::")
            if len(parts) != 3 or not parts[0].endswith(".py"):
                return False
            names.append("%s.%s.%s" % (parts[0][:-3].replace("/", "."), parts[1], parts[2]))

        def run():
            import unittest as ut

            suite = ut.TestLoader().loadTestsFromNames(names)
            result = ut.TestResult()
            result.failfast = True  # the first failure decides; the rest need not run
            suite.run(result)
            return result.testsRun == len(names) and result.wasSuccessful() and not result.skipped \
                and not result.expectedFailures and not result.unexpectedSuccesses

        ok = self.in_patched_child(run, timeout=min(CHILD_SECONDS, 3.0 + 3 * total / 1000))
        log("%d selected dialect tests re-run here:" % len(names), "all pass - none scheduled" if ok else "kept")
        return ok

    def max_nodes_recheck(self):
        """True if test_max_nodes, run on scaled copies of its input over the patched sources,
        gives exactly what it gives at the pinned commit (see max_nodes_check.py)."""
        if self.mx_ref is None or any(path.startswith("tests/") for path in self.changed):
            return False
        script = os.path.join(HERE, "max_nodes_check.py")
        reference = self.mx_ref

        def same():
            import importlib.util

            spec = importlib.util.spec_from_file_location("max_nodes_check", script)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.same_as(reference, module.observe(self.sources))

        ok = self.in_patched_child(same, timeout=min(CHILD_SECONDS, 10.0))
        log("test_max_nodes on scaled input:", "same as at the pinned commit - left out" if ok else "differs; kept")
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
                if hasattr(os, "fork") and self.forked_check(overlay, wanted, timeout=IMPORT_SECONDS):
                    return True
            except TimeoutError:
                # an import that hangs may stall the whole run (test collection imports it)
                self.import_hung = True
                return False
            except Exception as error:  # noqa: BLE001
                log("forked import check unavailable, using a fresh interpreter:", repr(error))
            if self.child_seconds >= TOTAL_CHILD_SECONDS:
                return False
            return self.fresh_check(overlay, wanted)
        except Exception as error:  # noqa: BLE001
            log("import check failed to run:", repr(error))
            return False
        finally:
            shutil.rmtree(overlay, ignore_errors=True)

    def in_patched_child(self, check, timeout=CHILD_SECONDS, snapshot=None):
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
            return self.forked_check(overlay, wanted, check, timeout, snapshot)
        except Exception as error:  # noqa: BLE001
            log("patched child failed to run:", repr(error))
            return False
        finally:
            shutil.rmtree(overlay, ignore_errors=True)

    def forked_check(self, overlay, wanted, check=None, timeout=CHILD_SECONDS, snapshot=None):
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
        limit = min(timeout, TOTAL_CHILD_SECONDS - self.child_seconds)
        if limit <= 0.5:
            raise TimeoutError("no time left for patched children")
        sys.stdout.flush()
        sys.stderr.flush()
        pid = os.fork()
        if pid == 0:
            status = 1
            try:
                import resource

                cpu = int(limit) + 1
                resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
                os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
                sys.stdout = sys.stderr
                import importlib

                before = snapshot() if snapshot is not None else None
                scope = {_module_of(p) for p in self.patched} | wanted
                # a package imported again must not keep stale submodules around (they would be
                # missing as attributes of the new package object), so those go too
                purged = set()
                for name in list(sys.modules):
                    if name in scope or any(name.startswith(pkg + ".") for pkg in scope):
                        del sys.modules[name]
                        purged.add(name)
                sys.meta_path.insert(0, _overlay_finder(overlay, self.sources))
                eager = sorted(wanted)
                if check is not None and "sqlglot.dialects.dialect" in purged:
                    # The dialect registry is new as well, so nothing stale can be looked up by
                    # name: whatever the check needs is imported, patched, when first used.
                    # (An import check has to import everything itself.)
                    eager = sorted(_module_of(p) for p in self.patched)
                for name in ["sqlglot"] + eager:
                    if name not in self.baseline():
                        importlib.import_module(name)
                status = 0 if check is None or (check(before) if snapshot is not None else check()) else 1
            except BaseException as error:  # noqa: BLE001
                if check is not None:
                    log("patched child:", repr(error)[:300])
                status = 1
            finally:
                os._exit(status)
        return self.reap(pid, limit)

    def reap(self, pid, limit):
        """Waits for a child for at most ``limit`` seconds; a child still running then (a
        patched loop that never ends, say) is killed and counts as a failure."""
        import signal
        import time

        start = time.monotonic()
        try:
            while True:
                done, code = os.waitpid(pid, os.WNOHANG)
                if done:
                    return os.WIFEXITED(code) and os.WEXITSTATUS(code) == 0
                if time.monotonic() - start > limit:
                    os.kill(pid, signal.SIGKILL)
                    os.waitpid(pid, 0)
                    log("patched child ran past %.0f s; stopped" % limit)
                    raise TimeoutError("patched child timed out")
                time.sleep(0.002)
        finally:
            self.child_seconds += time.monotonic() - start

    def fresh_check(self, overlay, wanted):
        import subprocess

        done = subprocess.run(
            [
                sys.executable, "-S", "-B", "-c", IMPORT_CHECK, self.sources, overlay,
                ",".join(self.baseline()), ",".join(sorted(wanted)),
            ],
            capture_output=True,
            text=True,
            timeout=max(1.0, min(30.0, TOTAL_CHILD_SECONDS - self.child_seconds)),
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

    _checking = False

    def _mark(self, acc, why):
        """Every test runs; ``why`` (kept for the log) says which rule gave up."""
        acc[1] = True
        if self.why is None:
            self.why = "check" if self._checking else why

    def _charge_label(self):
        import linecache

        frame = sys._getframe(2)
        name = frame.f_code.co_name
        text = linecache.getline(frame.f_code.co_filename, frame.f_lineno)
        if name == "charge_line":
            return "line"
        if name == "python_change":
            if "hit[0]" in text:
                return "line"
            if "source_readers" in text:
                return "file"
            return "value"
        if name in ("unimportable", "module_level"):
            return "check"
        return "file"

    def charge(self, acc, sid, queries=None, back=None, steps=0):
        """Adds a test set. ``queries`` says which queries of the per-query test the edit can
        reach through this set; without it, that test is charged as a whole. ``back`` says
        which of those it can reach after their plan exists (all of them when not known), and
        ``steps`` in which steps up to the plan (tpcds_check.STEPS; 0 when not known)."""
        if sid == GLOBAL:
            self._mark(acc, self._charge_label())
        elif sid >= 0:
            bits = self.sets[sid]
            acc[0] |= bits
            if self.pq_bit and bits & self.pq_bit:
                if queries is None:
                    acc[3] = True
                else:
                    acc[2].update(queries)
                    acc[4].update(queries if back is None else back)
                    acc[5] |= steps if steps else tpcds_check.STEP_BIT["pre"]

    def file_change(self, ch, acc):
        path = ch.path or ch.old_path
        self.touched.update(p for p in (ch.path, ch.old_path) if p)
        if path is None or ch.status == "unknown":
            self._mark(acc, "file")
            return
        base = path.rsplit("/", 1)[-1]
        if base in CONFIG_FILES:
            self._mark(acc, "file")
            return
        if path.endswith(".py"):
            info = self.files.get(path)
            if ch.status in ("delete", "rename") and info is not None:
                self.unimportable(path, info, acc)  # gone from where its importers look for it
                return
            if ch.status in ("delete", "rename", "binary") or (info is None and ch.status != "new"):
                self.module_level(path, info, acc)
                return
            if info is None:  # a new module nothing imports yet
                if path.startswith("sqlglot/"):
                    self._mark(acc, "file")
                return
            self.python_change(path, info, ch.blocks, acc)
            return
        self.data_change(path, ch, acc)

    def data_change(self, path, ch, acc):
        narrow = self.class_data_answer(path, ch)
        self.charge(acc, self.reads.get(path, NO_TESTS) if narrow is None else narrow)
        if ch.old_path and ch.old_path != path:
            self.charge(acc, self.reads.get(ch.old_path, NO_TESTS))
        if ch.status in ("new", "delete", "rename"):
            parts = path.split("/")
            for i in range(1, len(parts)):
                self.charge(acc, self.reads.get("/".join(parts[:i]) + "/", NO_TESTS))

    def class_data_answer(self, path, ch):
        """For a SQL fixture a test class's setUpClass loads into a list: the tests that read it
        outside such class fixtures plus those that used the list, provided the patched file
        still loads (otherwise setUpClass fails and the whole class with it). The per-query test
        is left out when no query's meta or SQL changed, the only parts of a pair it uses.
        None when that cannot be told."""
        entry = self.class_data.get(path)
        if entry is None or ch.status != "modify" or not ch.hunked or not ch.blocks:
            return None
        loader, sid, sid_without_pq, digest = entry
        try:
            with open(os.path.join(self.sources, path), "rb") as f:
                raw = f.read()
            if self.sha1(raw) != digest:
                return None
            old = raw.decode("utf-8").split("\n")
            for b in ch.blocks:
                if old[b.start - 1 : b.start - 1 + len(b.old)] != b.old:
                    return None
            new = _apply(old, ch.blocks, 1, len(old))
            import importlib.util
            import shutil
            import tempfile

            spec = importlib.util.spec_from_file_location("_fixture_helpers", os.path.join(self.sources, "tests", "helpers.py"))
            helpers = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helpers)
            work = tempfile.mkdtemp(prefix="blast-fixture-")
            try:
                for name, lines in (("old.sql", old), ("new.sql", new)):
                    with open(os.path.join(work, name), "w", encoding="utf-8") as f:
                        f.write("\n".join(lines))
                helpers.FIXTURES_DIR = work
                before = list(getattr(helpers, loader)("old.sql"))
                try:
                    after = list(getattr(helpers, loader)("new.sql"))
                except Exception:  # noqa: BLE001
                    log(path, "no longer loads; its test class fails as a whole")
                    return None
            finally:
                shutil.rmtree(work, ignore_errors=True)
        except Exception as error:  # noqa: BLE001
            log(path, "could not be compared:", repr(error))
            return None
        if self.pq_bit and sid >= 0 and self.sets[sid] & self.pq_bit and loader == "load_sql_fixture_pairs":
            if len(before) == len(after) and all(a[:2] == b[:2] for a, b in zip(before, after)):
                log(path, "changes no query the per-query test runs")
                return sid_without_pq
        return sid

    def module_level(self, path, info, acc):
        """An edit that runs when the module is imported."""
        if path.startswith("tests/"):
            sid = self.scopes.get(path)
            if sid is None:
                self._mark(acc, "file")
            else:
                self.charge(acc, sid)
        elif path.startswith("sqlglot/") or info is None:
            caller = sys._getframe(1).f_code.co_name
            self._mark(acc, {"unmodeled_edit": "unmodeled", "unimportable": "check"}.get(caller, "file"))
        else:
            self.charge(acc, info["users"])

    def scoped_edit(self, path, info, b, i, ln, acc):
        """An edit the line rules cannot place, narrowed by the compile unit around it (a
        module-level statement, or one statement of a class body). A blank or comment-only line
        removed (or left without code) changes nothing. When the unit is a def whose header
        (decorators, parameters, return annotation) stays the same, only runs of that function
        can see the edit: the tests that ran any line of it are charged, with an import check
        where it also ran at import. Any other edit to a class-body statement charges the users
        of the class and its family (from the map), with an import check. The rest is left to
        the caller."""
        flags = info["flags"]
        if not flags[ln - 1] & (IN_STRING | CONTINUED):
            old_text = b.old[i] if i < len(b.old) else None
            new_text = b.new[i] if len(b.old) == len(b.new) and i < len(b.new) else None
            if old_text is not None and not code_lines([old_text]) and (new_text is None or not code_lines([new_text])):
                return True
        lines = self.source_lines(path, info)
        units = self.units_for(info, [b])
        if lines is None or not units:
            return False
        (u0, u1), inner = next(iter(units.items()))
        try:
            old = ast.parse("\n".join(_dedent(lines[u0 - 1 : u1]) or [])).body
            new = ast.parse("\n".join(_dedent(_apply(lines, inner, u0, u1)) or [])).body
        except (SyntaxError, ValueError, TypeError):
            return False
        defs = (ast.FunctionDef, ast.AsyncFunctionDef)
        if len(old) == 1 and len(new) == 1 and isinstance(old[0], defs) and type(old[0]) is type(new[0]):
            dump = lambda n: ast.dump(n, include_attributes=False) if n is not None else ""
            head = lambda f: (f.name, dump(f.args), [dump(d) for d in f.decorator_list], dump(f.returns),
                              [dump(t) for t in getattr(f, "type_params", [])])
            if head(old[0]) == head(new[0]):
                log(path, "edit inside the body of %s; charging the tests that ran it" % old[0].name)
                for l2 in range(u0, u1 + 1):
                    self.charge_line(acc, info, l2)  # with the queries of the per-query test
                    if l2 in info["checks"]:
                        self.import_check.add(path)
                return True
        if not path.startswith("sqlglot/"):
            return False
        for a, z, sid in info.get("class_fb", ()):  # innermost class span first
            if a <= u0 and u1 <= z:
                if sid == GLOBAL:
                    return False
                log(path, "edit in a class body at line %d; charging the class's users" % u0)
                self.charge(acc, sid)
                self.import_check.add(path)
                return True
        return False

    def measured_change(self, acc):
        """Import-time edits the line rules cannot place, charged by what the patched import
        actually changed (see state_diff.py): readers of each changed class attribute or module
        global (by key for a dict), the tests that ran a changed function. Only when every
        top-level statement the patch touches can change nothing but its module's own state.
        False when that cannot be done; the caller then runs everything."""
        if self.global_readers is None or not hasattr(os, "fork"):
            return False
        scope = set()
        for path in self.unmodeled:  # the other files' edits were charged line by line
            info = self.files.get(path)
            if info is None or not path.startswith("sqlglot/") or not self.module_level_safe(path, info):
                return False
            scope.add(_module_of(path))
            scope.update(m for m in info.get("importers", ()) if not m.endswith("__main__"))
        if len(scope) > STATE_DIFF_MODULES:
            log("state diff: %d modules to compare, too many" % len(scope))
            return False
        names = sorted(scope)
        import pickle
        import tempfile

        handle, out_path = tempfile.mkstemp(prefix="blast-state-")
        os.close(handle)

        def check():
            import importlib

            for name in names:
                importlib.import_module(name)
            with open(out_path, "wb") as f:
                pickle.dump(state_diff.fingerprint(names, tpcds_check.value_digest), f)
            return True

        try:
            if not self.in_patched_child(check, timeout=STATE_DIFF_SECONDS):
                return False
            with open(out_path, "rb") as f:
                after = pickle.load(f)
        except Exception as error:  # noqa: BLE001
            log("state diff failed:", repr(error))
            return False
        finally:
            os.unlink(out_path)
        before = self.pristine_fingerprint(names)
        if before is None:
            return False
        diff = state_diff.differences(before, after)
        charged = 0
        for key, how in sorted(diff.items(), key=repr):
            if key[0] == "k":
                continue
            sid = self.state_charge(key, how, diff.get(("k",) + key))
            if isinstance(sid, tuple):  # a changed function: the lines it ran, with their queries
                path, a, z = sid
                for ln in range(a, z + 1):
                    self.charge_line(acc, self.files[path], ln)
                charged += 1
                continue
            if sid == GLOBAL:
                log("state diff: %s %s reaches every test" % (how, key[1:]))
                return False
            qs = self.attr_queries.get("%s:%s" % key[1:]) if key[0] == "a" and how == "changed" else None
            if qs is not None:
                queries = tuple(qs[0])
                self.charge(acc, sid, queries, self.back_of_known(queries, qs[1]), qs[2])
            else:
                self.charge(acc, sid)
            charged += 1
        log("state diff over %d modules: %d values changed" % (len(names), charged))
        return True

    def pristine_fingerprint(self, names):
        """The pinned import's fingerprints for these modules (cached per module)."""
        import importlib

        out = {}
        for name in names:
            got = self._pristine.get(name)
            if got is None:
                try:
                    importlib.import_module(name)
                    got = state_diff.fingerprint([name], tpcds_check.value_digest)
                except Exception as error:  # noqa: BLE001
                    log("pristine fingerprint of", name, "failed:", repr(error))
                    return None
                self._pristine[name] = got
            out.update(got)
        return out

    def state_charge(self, key, how, keys):
        """The test set for one changed module global or class attribute."""
        kind, where, attr = key
        value = None
        holder = sys.modules.get(where) if kind == "g" else self.pristine_class(where)
        if holder is not None:
            value = vars(holder).get(attr)
        func = value.__func__ if isinstance(value, (staticmethod, classmethod)) else value
        if isinstance(func, property):
            func = func.fget
        if isinstance(func, types.FunctionType) and how != "added":
            lines = self.def_lines(func)
            if lines is not None:
                path, a, z = lines
                if any(self.files[path]["sets"].get(ln) == GLOBAL for ln in range(a, z + 1)):
                    return GLOBAL
                return lines
        if how == "added":
            if kind == "g":
                return NO_TESTS  # a new module global: only new code can name it
            cls = holder
            for base in (cls.__mro__[1:] if cls is not None else ()):
                if attr in vars(base):
                    return self.tag_readers("%s.%s:%s" % (base.__module__, base.__qualname__, attr), None)
            return NO_TESTS  # a new class attribute nothing inherited: only new code can name it
        if kind == "g":
            tag = "%s:%s" % (where, attr)
            if how == "removed" and not self.still_named(where, attr):
                return self.global_external.get(tag, NO_TESTS)  # its module no longer uses it
            return self.global_readers.get(tag, NO_TESTS)
        return self.tag_readers("%s:%s" % (where, attr), keys if how == "changed" else None)

    def still_named(self, module, name):
        """True if the patched module still reads ``name`` as a bare name."""
        for path, (info, blocks) in self.changed.items():
            if _module_of(path) != module:
                continue
            try:
                tree = ast.parse(self.patched_text(path, info, blocks))
            except (SyntaxError, ValueError):
                return True
            return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(tree))
        return True

    def tag_readers(self, tag, keys):
        """Readers of a class attribute; by key when only some keys of a dict changed."""
        if keys is not None and tag in self.key_readers:
            base, per_key = self.key_readers[tag]
            if base == GLOBAL:
                return GLOBAL
            bits = self.sets[base] if base >= 0 else 0
            for k in keys:
                sid = per_key.get(k, NO_TESTS)
                if sid == GLOBAL:
                    return GLOBAL
                if sid >= 0:
                    bits |= self.sets[sid]
            return self.bits_sid(bits)
        return self.attr_readers.get(tag, NO_TESTS)

    def bits_sid(self, bits):
        """A set id for a bitset built here (appended to the table)."""
        if not bits:
            return NO_TESTS
        self.sets.append(bits)
        return len(self.sets) - 1

    def pristine_class(self, owner):
        """The pinned class for "module.qualname", if the module is imported here."""
        module, _, qualname = owner.partition(".")
        while True:
            obj = sys.modules.get(module)
            if obj is not None:
                for part in qualname.split("."):
                    obj = getattr(obj, part, None) if obj is not None else None
                if isinstance(obj, type):
                    return obj
            head, _, rest = qualname.partition(".")
            if not rest:
                return None
            module, qualname = module + "." + head, rest

    def def_lines(self, func):
        """(path, first line, last line) of a function's def in the pinned sources."""
        filename = func.__code__.co_filename
        if not filename.startswith(self.sources + os.sep):
            return None
        path = os.path.relpath(filename, self.sources)
        info = self.files.get(path)
        tree = self.old_tree(path, info) if info is not None else None
        if tree is None:
            return None
        qualname = func.__qualname__
        node = _defs(tree).get(qualname)
        if node is None or node.lineno > func.__code__.co_firstlineno or node.end_lineno < func.__code__.co_firstlineno:
            return None
        return path, node.lineno, node.end_lineno

    def old_tree(self, path, info):
        cache = self.__dict__.setdefault("_trees", {})
        if path not in cache:
            lines = self.source_lines(path, info)
            try:
                cache[path] = ast.parse("\n".join(lines)) if lines is not None else None
            except (SyntaxError, ValueError):
                cache[path] = None
        return cache[path]

    def module_level_safe(self, path, info):
        """True if every top-level statement the patch adds, removes or changes can change
        nothing but the module's own namespace (defs, classes, assignments to its names,
        imports, docstrings, and if/try blocks of those), so that fingerprints see all of it."""
        lines = self.source_lines(path, info)
        blocks = self.changed.get(path, (None, None))[1]
        if lines is None or blocks is None:
            return False
        try:
            old = ast.parse("\n".join(lines))
            new = ast.parse(self.patched_text(path, info, blocks))
        except (SyntaxError, ValueError):
            return False
        dump = lambda n: ast.dump(n, include_attributes=False)
        before = [dump(x) for x in old.body]
        after = [dump(x) for x in new.body]
        bound = {t.id for x in old.body + new.body if isinstance(x, (ast.Assign, ast.AnnAssign))
                 for t in (x.targets if isinstance(x, ast.Assign) else [x.target]) if isinstance(t, ast.Name)}
        import difflib

        matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
        for op, a0, a1, b0, b1 in matcher.get_opcodes():
            if op == "equal":
                continue
            for stmt in old.body[a0:a1] + new.body[b0:b1]:
                if not _own_state(stmt, bound):
                    return False
        return True

    def unmodeled_edit(self, path, info, acc):
        """An edit to code that runs when the module is imported, of a kind nothing narrower
        covers. If the module then no longer imports, what fails is the tests that use it
        (see unimportable); if it still imports, anything could have changed, so run()
        charges everything. The import check decides which."""
        fb = info.get("module_fb") if path.startswith("sqlglot/") else None
        if fb is None or not os.path.isdir(self.sources):
            self.module_level(path, info, acc)
            return
        # if it still imports, what it changed is measured (measured_change)
        self.unmodeled.add(path)
        self.import_check.add(path)

    def unimportable(self, path, info, acc):
        """A module that can no longer be imported fails every test that uses it or anything
        importing it: the tests that ran their code, created or looked into their classes, or
        import them (the map's "module_fb"). For a module that ``import sqlglot`` itself pulls
        in, that is every test."""
        fb = info.get("module_fb") if info is not None and path.startswith("sqlglot/") else None
        if fb is None or fb == GLOBAL:
            self.module_level(path, info, acc)
        else:
            log(path, "can no longer be imported; charging the tests that use it")
            self.charge(acc, fb)

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

    def binding_changes(self, path, info, blocks, acc):
        """Assigning to a name makes it local to the whole function, so ``x = []`` turned into
        ``exp = []`` breaks every read of ``exp`` in that function, wherever the edit sits.
        If an edit changes which names a function binds (or declares global/nonlocal), every
        test that ran any line of that function is charged."""
        lines = self.source_lines(path, info)
        units = self.units_for(info, blocks)
        if lines is None or units is None:
            return
        for (u0, u1), inner in units.items():
            old = _dedent(lines[u0 - 1 : u1])
            new = _dedent(_apply(lines, inner, u0, u1))
            if old is None or new is None:
                continue
            try:
                same = _bindings(old) == _bindings(new)
            except (SyntaxError, ValueError):
                same = False
            if same:
                continue
            log(path, "changes which names a function binds; charging the whole function")
            for ln in range(u0, u1 + 1):
                self.charge_line(acc, info, ln)  # with the queries of the per-query test
                if ln in info["checks"]:
                    self.import_check.add(path)

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

    def charge_line(self, acc, info, ln):
        """Charges the tests of a function line, with its queries of the per-query test. A line
        that also ran while sqlglot was imported has them only on condition that what the
        import computed with it comes out the same; that is checked with the queries."""
        sets, qsets = info["sets"], info.get("qsets", {})
        queries = qsets.get(ln)
        if queries is None and self.plans is not None and ln in info.get("qrt", {}):
            queries, after, values = info["qrt"][ln]
            self.import_values.update(tuple(v) for v in values)
            self.charge(acc, sets.get(ln, NO_TESTS), queries, self.back_of_known(queries, after),
                        info.get("stg", {}).get(ln, 0))
            return
        self.charge(acc, sets.get(ln, NO_TESTS), queries, self.back_of(info, ln, queries), info.get("stg", {}).get(ln, 0))

    def back_of_known(self, queries, after):
        """Which of ``queries`` the map says run ``after`` planning; None when unknown."""
        if queries is None or after is None or self.plans is None:
            return None
        return tuple(q for q in queries if q in set(after))

    def back_of(self, info, ln, queries):
        """Which of ``queries`` reach line ``ln`` after their plan exists; None when unknown."""
        if queries is None or self.plans is None:
            return None
        after = info.get("xq", {}).get(ln, ())
        return tuple(q for q in queries if q in after)

    def python_change(self, path, info, blocks, acc):
        kind = info["kind"]
        n = len(kind)
        self.changed[path] = (info, blocks)
        live = [b for b in blocks if not self.inert(info, b) and not self.pure_duplicate(path, info, b)]
        if not live:
            return
        # tests that read this file's source text (test_typing parses the typing modules' imports
        # out of it) see the edit
        self.charge(acc, self.source_readers.get(path, NO_TESTS))
        if self.source_lines(path, info) is None:
            # no verified pinned source: nothing can be compiled or imported to back a narrow
            # answer, so treat the edit as one that runs when the module is imported
            log(path, "has no verified source to check against")
            self.module_level(path, info, acc)
            return
        if not self.compiles(path, info, blocks):
            log(path, "does not compile after the patch")
            self.unimportable(path, info, acc)
            return
        sets = info["sets"]
        self.binding_changes(path, info, live, acc)
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
                        self.charge(acc, hit[0], hit[1], self.back_of(info, ln, hit[1]), info.get("stg", {}).get(ln, 0))
                        continue
                qsets = info.get("qsets", {})
                if k == RUNTIME:
                    self.charge_line(acc, info, ln)
                    if ln in info["checks"]:
                        self.import_check.add(path)
                    continue
                cols = None
                if pairs and b.old:
                    cols = changed_columns(b.old[i], b.new[i])
                if k == MIXED and cols is not None:
                    a, z = cols[0], max(cols[1], cols[0] + 1)
                    if any(c0 <= a and z <= c1 for c0, c1 in info["cols"].get(ln, ())):
                        self.charge_line(acc, info, ln)
                        if ln in info["checks"]:
                            self.import_check.add(path)
                        continue
                if k == DOC and cols is not None and cols[2].count('"') == cols[3].count('"') and cols[2].count("'") == cols[3].count("'"):
                    continue
                if k == IMPORT and cols is not None and ln in info["argt"]:
                    if SIMPLE_VALUE_RE.fullmatch(cols[2].strip()) and SIMPLE_VALUE_RE.fullmatch(cols[3].strip()):
                        # (expressions sqlglot builds while it is imported count too: an arg
                        # missing in one of those makes the set everything)
                        self.charge(acc, info["argt"][ln])
                        continue
                if k in (IMPORT, MIXED) and cols is not None and ln in info["dflt"]:
                    a, z = cols[0], max(cols[1], cols[0] + 1)
                    # (c0, c1, set, check[, queries of the per-query test, those after planning])
                    hit = [e[2:] for e in info["dflt"][ln] if e[0] <= a and z <= e[1]]
                    if hit:
                        for sid, check, *qs in hit:
                            queries = qs[0] if qs else None
                            self.charge(acc, sid, queries, self.back_of_known(queries, qs[1] if qs else None),
                                        qs[2] if len(qs) > 2 else 0)
                            # defaults are evaluated at import: a new name or call may fail there
                            if check or not LITERAL_RE.fullmatch(cols[3]):
                                self.import_check.add(path)
                        continue
                if k in (IMPORT, MIXED) and cols is not None and ln in info["cattr"]:
                    target = ASSIGN_TARGET_RE.match(b.old[i])
                    # (set, check[, queries of the per-query test, those after planning, steps])
                    sid, check, *qs = info["cattr"][ln]
                    queries = tuple(qs[0]) if qs else None
                    after = self.back_of_known(queries, qs[1]) if qs else None
                    steps = qs[2] if qs else 0
                    if target is None or cols[0] >= target.end():
                        a, z = cols[0], max(cols[1], cols[0] + 1)
                        entry = [e for e in info.get("kattr", {}).get(ln, ()) if e[0] <= a and z <= e[1]]
                        if entry:
                            sid = entry[0][2]  # the value of one dict entry: whoever looked up its key
                            if len(entry[0]) > 3:
                                queries = tuple(entry[0][3])
                                after = self.back_of_known(queries, entry[0][4])
                                steps = entry[0][5]
                            else:
                                queries, after, steps = None, None, 0
                        self.charge(acc, sid, queries, after, steps)  # the value changed
                        # whether import-time code can fail depends on the new value, not the old
                        if check or not LITERAL_RE.fullmatch(cols[3]):
                            self.import_check.add(path)
                        continue
                    renamed = ASSIGN_TARGET_RE.match(b.new[i])
                    if renamed is None or renamed.group(1) == target.group(1):
                        # the assignment was dropped or rewritten in place: same attribute
                        self.charge(acc, sid, queries, after, steps)
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
                    sid, _check, *qs = info["cattr"][ln]
                    queries = tuple(qs[0]) if qs else None
                    self.charge(acc, sid, queries, self.back_of_known(queries, qs[1]) if qs else None, qs[2] if qs else 0)
                    self.import_check.add(path)
                    continue
                if self.scoped_edit(path, info, b, i, ln, acc):
                    continue
                self.unmodeled_edit(path, info, acc)
            if acc[1]:
                return


# ----------------------------------------------------------------------------- packing
def _cost(workers, ms):
    workers = [w for w in workers if w]
    return max(sum(ms[t] for t in w) for w in workers) + WORKER_MS * len(workers) if workers else 0


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


def load_map(data):
    """The map blob: marshal data, xz- or zlib-compressed."""
    if data[:6] == b"\xfd7zXZ\x00":
        import lzma

        return marshal.loads(lzma.decompress(data))
    return marshal.loads(zlib.decompress(data))


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
            blob = load_map(f.read())
        number_of = {t["id"]: t["number"] for t in tests}
        index_to_number = [number_of.get(t, -1) for t in blob["tests"]]
        mapped = set(index_to_number)
        unknown = [t for t in every if t not in mapped]
        if unknown:
            log(len(unknown), "tests are not in the map; they run on every commit")
        selector = Selector(blob, SOURCE_ROOT)
        selector.ms = ms
        selector.index_to_number = index_to_number
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
