"""Builds the selector's map from a profile of the sqlglot suite at the pinned commit.

Usage (inside the profiling image, with the sqlglot checkout mounted at /src):

    python profiling/build_map.py --src /src --out data/map.bin \
        --profile out/main.txt.gz out/max_nodes.txt.gz --outcomes out/main.json out/max_nodes.json

The map is a zlib-compressed marshal blob. Test sets are stored as Python ints used as
bitsets over test indices; ``GLOBAL`` (-2) means "every test".
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import marshal
import os
import re
import sys
import zlib
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from source_index import DOC, IMPORT, MIXED, NONE, RUNTIME, index_source  # noqa: E402

NO_TESTS = -1
GLOBAL = -2
FORMAT = 1
PINNED = "eb11aec4be8cb2122afe713a4816662c3f0fb2fa"
SOURCE_DIRS = ("sqlglot/", "tests/", "sqlglotc/")


# ----------------------------------------------------------------------------- profile
def read_profile(paths):
    """ctx -> {"lines": {(rel, signed line)}, "reads": {path}, "missing": {(class, arg)}}"""
    contexts = defaultdict(lambda: {"lines": set(), "reads": set(), "missing": set()})
    for path in paths:
        files = {}
        records = []
        with gzip.open(path, "rt") as f:
            for raw in f:
                kind, rest = raw[0], raw[2:].rstrip("\n")
                if kind == "F":
                    fid, rel = rest.split("\t", 1)
                    files[int(fid)] = rel
                else:
                    records.append((kind, rest))
        for kind, rest in records:
            ctx, _, payload = rest.partition("\t")
            entry = contexts[ctx]
            if kind == "C" and payload:
                for item in payload.split(","):
                    fid, line = item.split(":")
                    entry["lines"].add((files[int(fid)], int(line)))
            elif kind == "O" and payload:
                entry["reads"].update(payload.split("|"))
            elif kind == "V" and payload:
                for item in payload.split(","):
                    cls, _, arg = item.rpartition(":")
                    entry["missing"].add((cls, arg))
    return contexts


def read_outcomes(paths):
    order, failed = [], []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        for nodeid, result in data["results"].items():
            order.append(nodeid)
            if result["failed"]:
                failed.append(nodeid)
        if data["collect_errors"]:
            failed.extend(data["collect_errors"])
    return order, failed


class Tests:
    def __init__(self, nodeids):
        self.ids = list(dict.fromkeys(nodeids))
        self.index = {t: i for i, t in enumerate(self.ids)}
        self._scope_cache = {}

    def scope(self, node):
        """Bitset of tests at or below a collection node id."""
        if node in self._scope_cache:
            return self._scope_cache[node]
        bits = 0
        prefix = node + "::"
        for t, i in self.index.items():
            if t == node or t.startswith(prefix):
                bits |= 1 << i
        self._scope_cache[node] = bits
        return bits

    def resolve(self, ctx):
        """Tests charged for something that happened in this context, or GLOBAL."""
        if ctx in ("<import>", "<other>"):
            return GLOBAL
        if ctx.startswith("<collect>|"):
            return self.scope(ctx[len("<collect>|"):])
        if ctx.startswith("<fixture>|"):
            _, _scope, node = ctx.split("|", 2)
            return GLOBAL if node == "" else self.scope(node)
        nodeid, _, _phase = ctx.rpartition("|")
        i = self.index.get(nodeid)
        return GLOBAL if i is None else 1 << i


class SetTable:
    def __init__(self):
        self.sets = []
        self.ids = {}

    def add(self, bits):
        if bits == GLOBAL:
            return GLOBAL
        if not bits:
            return NO_TESTS
        sid = self.ids.get(bits)
        if sid is None:
            sid = self.ids[bits] = len(self.sets)
            self.sets.append(bits)
        return sid


# ----------------------------------------------------------------------------- tests/ imports
_IMPORT_RE = re.compile(r"^\s*(?:from\s+([.\w]+)\s+import|import\s+([.\w, ]+))", re.M)


def test_module_scopes(src, tests):
    """For each module under tests/: bitset of tests in it and in every module importing it."""
    modules = {}
    for root, _dirs, names in os.walk(os.path.join(src, "tests")):
        for name in names:
            if name.endswith(".py"):
                rel = os.path.relpath(os.path.join(root, name), src)
                modules[rel] = open(os.path.join(src, rel), encoding="utf-8").read()

    def module_file(dotted, here):
        if dotted.startswith("."):
            pkg = os.path.dirname(here)
            level = len(dotted) - len(dotted.lstrip("."))
            for _ in range(level - 1):
                pkg = os.path.dirname(pkg)
            dotted = pkg.replace("/", ".") + ("." + dotted.lstrip(".") if dotted.lstrip(".") else "")
        base = dotted.replace(".", "/")
        for cand in (base + ".py", base + "/__init__.py"):
            if cand in modules:
                return cand
        return None

    importers = defaultdict(set)
    for rel, text in modules.items():
        deps = set()
        for m in _IMPORT_RE.finditer(text):
            names = [m.group(1)] if m.group(1) else [n.strip().split(" ")[0] for n in m.group(2).split(",")]
            for dotted in names:
                target = module_file(dotted, rel)
                if target:
                    deps.add(target)
        # importing a module runs its packages' __init__ first
        parts = rel.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            init = "/".join(parts[:i]) + "/__init__.py"
            if init in modules and init != rel:
                deps.add(init)
        for dep in deps:
            importers[dep].add(rel)

    scopes = {}
    for rel in modules:
        seen, stack = {rel}, [rel]
        while stack:
            for imp in importers[stack.pop()]:
                if imp not in seen:
                    seen.add(imp)
                    stack.append(imp)
        bits = 0
        for mod in seen:
            bits |= tests.scope(mod)
        scopes[rel] = bits
    return scopes


# ----------------------------------------------------------------------------- arg_types
def argtypes_graph(src, static):
    """Edges origin class -> classes whose arg_types (and so required_args) derive from it.

    A class inherits arg_types unless it assigns its own, and an own dict can splice in another
    class's with ``**Other.arg_types``. Classes are keyed "module.qualname", like the profile.
    """
    sys.path.insert(0, src)
    import importlib
    import pkgutil

    import sqlglot
    from sqlglot.expressions import core

    for info in pkgutil.walk_packages(sqlglot.__path__, "sqlglot."):
        if not info.name.endswith(".__main__"):
            try:
                importlib.import_module(info.name)
            except Exception as error:  # noqa: BLE001
                print("build_map: import %s failed: %r" % (info.name, error), file=sys.stderr)

    def key(cls):
        return "%s.%s" % (cls.__module__, cls.__qualname__)

    def rel_file(cls):
        mod = sys.modules.get(cls.__module__)
        path = getattr(mod, "__file__", "") or ""
        return os.path.relpath(os.path.abspath(path), src)

    classes, stack, seen = [], [core.Expr], set()
    while stack:
        cls = stack.pop()
        for sub in cls.__subclasses__():
            if sub not in seen:
                seen.add(sub)
                classes.append(sub)
                stack.append(sub)

    edges = defaultdict(set)
    by_site = {}
    opaque = []
    for cls in classes:
        k = key(cls)
        if "arg_types" in cls.__dict__:
            info = static.get((rel_file(cls), cls.__qualname__))
            if info is None:
                opaque.append(k)
                continue
            by_site[(rel_file(cls), cls.__qualname__)] = k
            module = sys.modules[cls.__module__]
            sources = list(info["spreads"]) + ([info["expr"]] if info["expr"] else [])
            for text in sources:
                base = re.sub(r"(\.arg_types|\.copy\(\))+$", "", text)
                obj = module.__dict__
                ok = bool(re.fullmatch(r"[A-Za-z_][\w.]*", base))
                if ok:
                    parts = base.split(".")
                    target = obj.get(parts[0])
                    for p in parts[1:]:
                        target = getattr(target, p, None)
                    if isinstance(target, type):
                        edges[key(target)].add(k)
                        continue
                    if isinstance(target, dict):
                        # A module-level dict spliced in: edits to it are module-level and run
                        # everything anyway; this class's own entries stay traceable.
                        continue
                opaque.append(k)
        else:
            for parent in cls.__mro__[1:]:
                if "arg_types" in parent.__dict__:
                    edges[key(parent)].add(k)
                    break
    return edges, by_site, set(opaque)


# ----------------------------------------------------------------------------- build
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--profile", nargs="+", required=True)
    ap.add_argument("--outcomes", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src = os.path.abspath(args.src)

    order, failed = read_outcomes(args.outcomes)
    if failed:
        print("build_map: %d tests failed during profiling: %s" % (len(failed), failed[:5]), file=sys.stderr)
    tests = Tests(order)
    contexts = read_profile(args.profile)
    table = SetTable()

    # line -> tests (function code only), and lines run by function code in GLOBAL contexts
    # A module's body (negative lines) running in a test's context means that test imported it.
    cov = defaultdict(int)
    global_func = set()
    module_users = defaultdict(int)
    module_global = set()
    for ctx, entry in contexts.items():
        charged = tests.resolve(ctx)
        for rel, line in entry["lines"]:
            if line > 0:
                if charged == GLOBAL:
                    global_func.add((rel, line))
                else:
                    cov[(rel, line)] |= charged
            elif charged == GLOBAL:
                module_global.add(rel)
            else:
                module_users[rel] |= charged
    for rel in module_global:
        module_users[rel] = GLOBAL

    files = {}
    static_argtypes = {}
    digests = {}
    for root, dirs, names in os.walk(src):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "sqlglot-integration-tests")]
        for name in names:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, src)
            with open(path, "rb") as f:
                raw = f.read()
            digests[rel] = hashlib.sha1(raw).hexdigest()
            try:
                ix = index_source(raw.decode("utf-8"))
            except (SyntaxError, UnicodeDecodeError) as error:
                print("build_map: cannot index %s: %r" % (rel, error), file=sys.stderr)
                continue
            for qualname, info in ix["argtypes_classes"].items():
                static_argtypes[(rel, qualname)] = info
            files[rel] = ix

    edges, by_site, opaque = argtypes_graph(src, static_argtypes)
    if opaque:
        print("build_map: %d classes with arg_types we cannot trace: %s" % (len(opaque), sorted(opaque)[:8]), file=sys.stderr)

    missing = defaultdict(int)
    for ctx, entry in contexts.items():
        charged = tests.resolve(ctx)
        for cls, arg in entry["missing"]:
            if charged == GLOBAL or missing[(cls, arg)] == GLOBAL:
                missing[(cls, arg)] = GLOBAL
            else:
                missing[(cls, arg)] |= charged

    def descendants(k):
        seen, stack = {k}, [k]
        while stack:
            for child in edges.get(stack.pop(), ()):
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        return seen

    out_files = {}
    for rel, ix in files.items():
        kind = ix["kind"]
        spans = ix["spans"]
        sets = {}
        for ln in range(1, ix["n"] + 1):
            k = kind[ln - 1]
            if k not in (RUNTIME, MIXED):
                continue
            a, b = spans[ln - 1]
            bits = 0
            glob = False
            for l2 in range(a, b + 1):
                bits |= cov.get((rel, l2), 0)
                if (rel, l2) in global_func:
                    glob = True
            sid = GLOBAL if glob else table.add(bits)
            if sid != NO_TESTS:
                sets[ln] = sid

        argt = {}
        for ln, entries in ix["argtypes"].items():
            bits = 0
            glob = False
            for qualname, arg in entries:
                origin = by_site.get((rel, qualname))
                if origin is None or origin in opaque:
                    glob = True
                    continue
                for cls in descendants(origin):
                    if cls in opaque:
                        glob = True
                    m = missing.get((cls, arg), 0)
                    if m == GLOBAL:
                        glob = True
                    else:
                        bits |= m
            argt[ln] = GLOBAL if glob else table.add(bits)

        users = module_users.get(rel, 0)
        out_files[rel] = {
            "kind": kind,
            "flags": ix["flags"],
            "sets": sets,
            "cols": ix["runtime_cols"],
            "argt": argt,
            "users": GLOBAL if users == GLOBAL else table.add(users),
            "sha1": digests[rel],
        }

    openers = defaultdict(int)
    for ctx, entry in contexts.items():
        charged = tests.resolve(ctx)
        for path in entry["reads"]:
            if "__pycache__" in path or (ctx == "<other>" and path.endswith("/")):
                continue  # bytecode caches, and pytest walking directories while collecting
            if charged == GLOBAL or openers[path] == GLOBAL:
                openers[path] = GLOBAL
            else:
                openers[path] |= charged

    scopes = test_module_scopes(src, tests)

    blob = {
        "format": FORMAT,
        "commit": PINNED,
        "tests": tests.ids,
        "files": out_files,
        "reads": {p: (GLOBAL if b == GLOBAL else table.add(b)) for p, b in openers.items()},
        "scopes": {p: table.add(b) for p, b in scopes.items()},
        "sets": table.sets,
    }
    data = zlib.compress(marshal.dumps(blob), 9)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(data)
    print(
        "build_map: %d tests, %d files, %d sets, %d read paths -> %s (%.1f MB)"
        % (len(tests.ids), len(out_files), len(table.sets), len(openers), args.out, len(data) / 1e6),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
