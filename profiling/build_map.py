"""Builds the selector's map from a profile of the sqlglot suite at the pinned commit.

Usage (inside the profiling image, with the sqlglot checkout mounted at /src):

    python profiling/build_map.py --src /src --out data/map.bin \
        --profile out/main.txt.gz out/max_nodes.txt.gz --outcomes out/main.json out/max_nodes.json

The map is a zlib-compressed marshal blob. Test sets are stored as Python ints used as
bitsets over test indices; ``GLOBAL`` (-2) means "every test".
"""

from __future__ import annotations

import argparse
import ast
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

from import_effects import MODULE, ImportEffects, module_name  # noqa: E402
from source_index import DOC, IMPORT, MIXED, NONE, RUNTIME, index_source  # noqa: E402

NO_TESTS = -1
GLOBAL = -2
FRESH_INTERPRETER_TESTS = ("tests/dialects/test_dialect.py::TestDialect::test_lazy_load",)
# Split into one profile context per query; the selector can re-check single queries of it.
PER_QUERY_TEST = "tests/test_executor.py::TestExecutor::test_execute_tpcds"
# The steps of each of its queries up to the plan; must match src/tpcds_check.py STEPS.
TPCDS_STEPS = ["parse", "pickle", "tables", "pre"] + ["rule%d" % i for i in range(14)] + ["plan", "duckdb"]
OUTSIDE = -1  # touched outside any single query
FORMAT = 1
PINNED = "eb11aec4be8cb2122afe713a4816662c3f0fb2fa"
SOURCE_DIRS = ("sqlglot/", "tests/", "sqlglotc/")


# ----------------------------------------------------------------------------- profile
def _split_runs(paths):
    """Profiles that split the per-query test into one context per query."""
    out = set()
    for path in paths:
        with gzip.open(path, "rt") as f:
            for raw in f:
                if raw[0] in "CW" and (PER_QUERY_TEST + "|call#q") in raw.split("\t", 1)[0]:
                    out.add(path)
                    break
    return out


def read_profile(paths, meta=None):
    """ctx -> {"lines": {(rel, signed line)}, "reads": {path}, "missing": {(class, arg)},
    "attrs": {"owner:attr"}}. Class metadata (A and K records) goes into ``meta``."""
    contexts = defaultdict(lambda: {"lines": set(), "reads": set(), "missing": set(), "attrs": set()})
    if meta is None:
        meta = {}
    meta.setdefault("instrumented", set())
    meta.setdefault("families", {})
    for path in paths:
        files = {}
        records = []
        with gzip.open(path, "rt") as f:
            for raw in f:
                kind, rest = raw[0], raw[2:].rstrip("\n") if raw[1:2] == " " else raw[1:].rstrip("\n")
                if kind == "F":
                    fid, rel = rest.split("\t", 1)
                    files[int(fid)] = rel
                else:
                    records.append((kind, rest))
        unsplit = path not in meta.get("split_runs", {path})
        for kind, rest in records:
            if unsplit and rest.startswith(PER_QUERY_TEST + "|call\t"):
                rest = PER_QUERY_TEST + "|call#unsplit" + rest[len(PER_QUERY_TEST + "|call"):]
            if kind == "S":
                meta["sites"] = json.loads(rest.lstrip("\t"))
                continue
            if kind == "A":
                owner, _, names = rest.partition("\t")
                meta["instrumented"].update((owner, n) for n in names.split(",") if n)
                continue
            if kind == "I":
                where, _, origins = rest.partition("\t")
                fid, line = where.split(":")
                dest = meta.setdefault("origins", {}).setdefault((files[int(fid)], int(line)), set())
                for item in origins.split(","):
                    ofid, oline = item.split(":")
                    dest.add((files[int(ofid)], int(oline)))
                continue
            if kind == "G":
                for site, entry in enumerate(json.loads(rest.lstrip("\t"))):
                    meta.setdefault("default_sites", {})[(path, site)] = tuple(entry)
                continue
            if kind == "P":
                tag, path, loader = (rest.split("\t") + ["", ""])[:3]
                meta.setdefault("provenance", {})[tag] = (path, loader)
                continue
            if kind == "K":
                key, mro, parts = (rest.split("\t") + ["", ""])[:3]
                meta["families"][key] = (tuple(m for m in mro.split("|") if m), tuple(p for p in parts.split("|") if p))
                continue
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
            elif kind == "R" and payload:
                entry["attrs"].update(payload.split(","))
            elif kind == "W" and payload:
                entry.setdefault("weak", set()).update(int(c) for c in payload.split(","))
            elif kind == "U" and payload:
                entry.setdefault("users", set()).update(payload.split(","))
            elif kind == "E" and payload:
                entry.setdefault("defaults", set()).update((path, int(x)) for x in payload.split(","))
            elif kind == "M" and payload:
                entry.setdefault("dyn", set()).update(payload.split(","))
            elif kind == "D" and payload:
                entry.setdefault("data", set()).update(payload.split(","))
            elif kind == "Y" and payload:
                entry.setdefault("keys", set()).update((tag, k) for tag, k in json.loads(payload))
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


def per_query_reference(path, index, staged=False):
    """Reference digests and timings for the per-query test, or None."""
    if not path or index is None:
        return None
    with open(path) as f:
        ref = json.load(f)
    return {
        "test": index,
        "load": ref["load_seconds"],
        "rows": [q["rows"] for q in ref["queries"]],
        "duckdb": [q["duckdb"] for q in ref["queries"]],
        "seconds": [q["seconds"] for q in ref["queries"]],
        "staged": staged,  # files carry "xq": the queries an edited line reaches after planning
    }


def import_time_instances():
    """Classes of the expression nodes that exist right after import: SQL templates parsed into
    module globals, class attributes, defaults and closures. An edit to such a class can change
    those nodes, which tests then use without ever creating one, so it is not charged to the
    class's users alone."""
    from sqlglot.expressions import core

    out, seen, stack = set(), set(), []
    for name, mod in list(sys.modules.items()):
        if name.split(".")[0] == "sqlglot" and mod is not None:
            stack.extend(vars(mod).values())
    while stack:
        value = stack.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, core.Expr):
            out.add("%s.%s" % (type(value).__module__, type(value).__qualname__))
            stack.extend(value.args.values())
        elif isinstance(value, type):
            if (value.__module__ or "").split(".")[0] == "sqlglot":
                stack.extend(vars(value).values())
        elif isinstance(value, (list, tuple, set, frozenset)):
            stack.extend(value)
        elif isinstance(value, dict):
            stack.extend(value.keys())
            stack.extend(value.values())
        elif isinstance(value, (staticmethod, classmethod)):
            stack.append(value.__func__)
        elif callable(value) and hasattr(value, "__code__"):
            stack.extend(value.__defaults__ or ())
            stack.extend((value.__kwdefaults__ or {}).values())
            for cell in value.__closure__ or ():
                try:
                    stack.append(cell.cell_contents)
                except ValueError:
                    pass
    return out


_DIALECT_ATTR = re.compile(r"dialects\.([A-Z][A-Za-z0-9]*)\b")


def test_module_imports(src, files, modules):
    """test file -> sqlglot modules it imports anywhere, including inside test methods and the
    lazy ``from sqlglot.dialects import MySQL`` / ``sqlglot.dialects.MySQL`` forms."""
    out = {}
    for rel, ix in files.items():
        if not rel.startswith("tests/"):
            continue
        found = set()
        for level, mod, names in ix.get("all_imports", ()):
            if level:
                continue
            for target in [mod] + ["%s.%s" % (mod, n) for n in names]:
                if target in modules:
                    found.add(target)
                if mod == "sqlglot.dialects":
                    for n in names:
                        if "sqlglot.dialects." + n.lower() in modules:
                            found.add("sqlglot.dialects." + n.lower())
        with open(os.path.join(src, rel), encoding="utf-8") as f:
            for n in _DIALECT_ATTR.findall(f.read()):
                if "sqlglot.dialects." + n.lower() in modules:
                    found.add("sqlglot.dialects." + n.lower())
        out[rel] = found
    return out


def core_modules(src):
    """Modules a fresh interpreter loads for a plain ``import sqlglot``."""
    import subprocess

    code = "import sys, sqlglot; print(','.join(sorted(m for m in sys.modules if m.split('.')[0] == 'sqlglot')))"
    env = dict(os.environ, PYTHONPATH=src)
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd="/", timeout=300)
    names = [m for m in done.stdout.strip().split(",") if m]
    if done.returncode != 0 or not names:
        raise SystemExit("build_map: could not list the modules 'import sqlglot' loads: %s" % done.stderr[-500:])
    return names


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
def _dict_owner(value, defined):
    """"module.NAME" of the module-level dict display that ``value`` is, found by identity
    among the modules holding it and the dict displays the source index saw."""
    for name, mod in list(sys.modules.items()):
        if name.split(".")[0] != "sqlglot" or mod is None:
            continue
        for attr, obj in list(vars(mod).items()):
            if obj is value and (name, attr) in defined:
                return "%s.%s" % (name, attr)
    return None


def key_only_uses(src, name, files):
    """True if every reference to the module-level dict ``name`` outside arg_types splices only
    looks at its keys: iterating it, or an ``in`` test against it."""
    import ast as _ast

    for rel, ix in files.items():
        if not rel.startswith("sqlglot/"):
            continue
        with open(os.path.join(src, rel), encoding="utf-8") as f:
            text = f.read()
        if name not in text:
            continue
        tree = _ast.parse(text)
        ok_nodes = set()
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.For, _ast.comprehension)):
                ok_nodes.add(id(node.iter))
            elif isinstance(node, _ast.Compare):
                for op, right in zip(node.ops, node.comparators):
                    if isinstance(op, (_ast.In, _ast.NotIn)):
                        ok_nodes.add(id(right))
            elif isinstance(node, _ast.Dict):
                for k, v in zip(node.keys, node.values):
                    if k is None:
                        ok_nodes.add(id(v))  # ** splice; arg_types splices are handled as arg_types
            elif isinstance(node, (_ast.Import, _ast.ImportFrom)):
                for alias in node.names:
                    if alias.name == name:
                        ok_nodes.add(id(alias))
            elif isinstance(node, _ast.Assign):
                for target in node.targets:
                    if isinstance(target, _ast.Name) and target.id == name:
                        ok_nodes.add(id(target))  # the definition itself
        for node in _ast.walk(tree):
            ref = (isinstance(node, _ast.Name) and node.id == name) or (isinstance(node, _ast.Attribute) and node.attr == name)
            if ref and id(node) not in ok_nodes:
                return False
    return True


def argtypes_graph(src, static, defined_dicts=frozenset()):
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
                        # A module-level dict spliced in (``**QUERY_MODIFIERS``): the class
                        # derives from it like from another class's arg_types.
                        owner = _dict_owner(target, defined_dicts)
                        if owner is not None:
                            edges["dict:" + owner].add(k)
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
    ap.add_argument("--weak", nargs="*", default=[], help="profiles recorded with PROFILE_WEAK=1")
    ap.add_argument("--weak-outcomes", nargs="*", default=[], help="outcomes of the weak runs")
    ap.add_argument("--keys", nargs="*", default=[], help="profiles recorded with PROFILE_KEYS=1 (also in --profile)")
    ap.add_argument("--tpcds", default=None, help="reference fingerprints from src/tpcds_check.py")
    ap.add_argument("--tpcds-stages", default=None, help="front/back lines per query from profiling/tpcds_stages.py")
    ap.add_argument("--defaults", nargs="*", default=[], help="profiles recorded with PROFILE_DEFAULTS=1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src = os.path.abspath(args.src)

    order, failed = read_outcomes(args.outcomes)
    if failed:
        print("build_map: %d tests failed during profiling: %s" % (len(failed), failed[:5]), file=sys.stderr)
    tests = Tests(order)
    meta = {"split_runs": _split_runs(args.profile)}
    contexts = read_profile(args.profile, meta)
    table = SetTable()

    # line -> tests (function code only), and lines run by function code in GLOBAL contexts
    # A module's body (negative lines) running in a test's context means that test imported it.
    cov = defaultdict(int)
    global_func = set()
    module_users = defaultdict(int)
    module_global = set()
    pq_index = tests.index.get(PER_QUERY_TEST)
    pq_bit = 1 << pq_index if pq_index is not None else 0
    pq_prefix = PER_QUERY_TEST + "|call#q"
    qcov = defaultdict(set)  # (rel, line) -> queries of the per-query test that ran it (or OUTSIDE)
    # Per query, which lines run before the plan exists (front) and which anywhere else (back);
    # an edit only the front reaches is re-checked by comparing plans instead of results.
    stage_front, stage_back = defaultdict(set), defaultdict(set)
    stage_modes = {}  # (rel, line) -> bitmask of the steps (tpcds_check.STEPS) it ran in
    if args.tpcds_stages and pq_index is not None:
        with open(args.tpcds_stages) as f:
            stages = json.load(f)
        bit = {name: 1 << i for i, name in enumerate(TPCDS_STEPS)}
        for rel, lines in stages.get("modes", {}).items():
            for ln, modes in lines.items():
                stage_modes[(rel, int(ln))] = sum(bit.get(m, bit["pre"]) for m in set(modes))
        for name, target in (("front", stage_front), ("back", stage_back)):
            for rel, lines in stages[name].items():
                for ln, queries in lines.items():
                    target[(rel, int(ln))].update(queries)
    for ctx, entry in contexts.items():
        charged = tests.resolve(ctx)
        query = int(ctx[len(pq_prefix):]) if ctx.startswith(pq_prefix) else None
        unsplit = ctx.endswith("|call#unsplit")
        for rel, line in entry["lines"]:
            if line > 0:
                if charged == GLOBAL:
                    global_func.add((rel, line))
                else:
                    cov[(rel, line)] |= charged
                    if charged & pq_bit:
                        if unsplit:  # a run that did not split the test only says "somewhere in it"
                            qcov[(rel, line)].add("unsplit")
                        else:
                            qcov[(rel, line)].add(OUTSIDE if query is None else query)
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

    defined_dicts = {(module_name(rel), name) for rel, ix in files.items() if rel.startswith("sqlglot/")
                     for qual, name in ix.get("dict_displays", []) if not qual}
    edges, by_site, opaque = argtypes_graph(src, static_argtypes, defined_dicts)
    spliced_ok = {origin: key_only_uses(src, origin.rsplit(".", 1)[1], files)
                  for origin in edges if origin.startswith("dict:")}
    print("build_map: module dicts spliced into arg_types: %s" % spliced_ok, file=sys.stderr)
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

    # ------------------------------------------------------------- import-time effects
    attr_reads = defaultdict(int)
    attr_global = set()
    for ctx, entry in contexts.items():
        charged = tests.resolve(ctx)
        for tag in entry["attrs"]:
            if charged == GLOBAL:
                attr_global.add(tag)
            else:
                attr_reads[tag] |= charged

    mentions = defaultdict(list)
    for rel, ix in files.items():
        for name, lines in ix["runtime_names"].items():
            mentions[name].append((rel, lines))
    name_cache = {}
    origins = meta.get("origins", {})
    eff_cache = {}
    eff_active = set()
    via_import = set()

    def eff(rel, ln):
        """Tests an edit to function line ``ln`` can reach: the tests that ran it, plus, when
        it also ran while sqlglot was being imported, the readers of every class attribute or
        module global it helped compute then (sqlglot parses SQL templates and builds tables
        at import time). GLOBAL when that import-time role cannot be traced."""
        key = (rel, ln)
        if key in eff_cache:
            return eff_cache[key]
        if key in eff_active:
            return GLOBAL
        eff_active.add(key)
        bits = cov.get(key, 0)
        if key in global_func:
            via_import.add(key)
            found = origins.get(key)
            if not found:
                bits = GLOBAL
            for orel, oline in found or ():
                entries = files[orel]["classattr"].get(oline) if orel in files and orel.startswith("sqlglot/") else None
                if not entries:
                    bits = GLOBAL
                    break
                for qualname, name in entries:
                    owner = "%s.%s" % (module_name(orel), qualname) if qualname else MODULE
                    got = effects.tests_for(owner, name)
                    if got == GLOBAL:
                        bits = GLOBAL
                        break
                    bits |= got
                if bits == GLOBAL:
                    break
        eff_active.discard(key)
        eff_cache[key] = bits
        return bits

    def import_values(rel, a, b):
        """The class attributes and module globals (module, class qualname or "", name) whose
        import-time computation ran any of lines a..b; None if one of those runs cannot be
        traced to an assignment."""
        out = set()
        for l2 in range(a, b + 1):
            if (rel, l2) not in via_import:
                continue
            found = origins.get((rel, l2))
            if not found:
                return None
            for orel, oline in found:
                entries = files[orel]["classattr"].get(oline) if orel in files and orel.startswith("sqlglot/") else None
                if not entries:
                    return None
                for qualname, name in entries:
                    out.add((module_name(orel), qualname or "", name))
        return tuple(sorted(out))

    def union_eff(rel, lines):
        bits = 0
        for ln in lines:
            got = eff(rel, ln)
            if got == GLOBAL:
                return GLOBAL
            bits |= got
        return bits

    def name_cov(name):
        """Tests that ran a statement mentioning ``name`` (readers of a module global)."""
        if name not in name_cache:
            name_cache[name] = GLOBAL  # guards against cycles while it is being computed
            bits = 0
            for rel, lines in mentions.get(name, ()):
                got = union_eff(rel, lines)
                if got == GLOBAL:
                    bits = GLOBAL
                    break
                bits |= got
            name_cache[name] = bits
        return name_cache[name]

    effects = ImportEffects(files, meta["families"], meta["instrumented"], attr_reads, attr_global, name_cov)

    # Dict keys looked up per test (PROFILE_KEYS=1). A test that read a dict attribute but left
    # no key record used it in a way the profile cannot see and is charged for every key.
    key_bits = defaultdict(int)
    key_tags = set()
    key_contexts = read_profile(args.keys, {"split_runs": _split_runs(args.keys)}) if args.keys else {}
    for ctx, entry in key_contexts.items():
        found = entry.get("keys")
        if not found:
            continue
        charged = tests.resolve(ctx)
        for tag, k in found:
            key_tags.add(tag)
            if charged == GLOBAL or key_bits[(tag, k)] == GLOBAL:
                key_bits[(tag, k)] = GLOBAL
            else:
                key_bits[(tag, k)] |= charged
    # Readers seen by the keyed runs themselves: one of those with no key record never looked
    # inside the dict (every way in is recorded). Readers only seen in other runs are charged.
    keyrun_reads = defaultdict(int)
    if args.keys:
        for ctx, entry in key_contexts.items():
            charged = tests.resolve(ctx)
            for tag in entry["attrs"]:
                if charged == GLOBAL or keyrun_reads[tag] == GLOBAL:
                    keyrun_reads[tag] = GLOBAL
                else:
                    keyrun_reads[tag] |= charged

    def keyed(tag, k):
        if tag not in key_tags and tag not in keyrun_reads:
            return attr_reads.get(tag, 0)  # never profiled by key: charge every reader
        parts = [key_bits.get((tag, k), 0), key_bits.get((tag, "*"), 0), keyrun_reads.get(tag, 0)]
        if GLOBAL in parts:
            return GLOBAL
        unseen = attr_reads.get(tag, 0) & ~parts[2]
        return parts[0] | parts[1] | unseen

    displays = set()
    for rel, ix in files.items():
        for qualname, attr in ix.get("dict_displays", ()):
            if qualname:
                displays.add(("%s.%s" % (module_name(rel), qualname), attr))
    if not meta["families"]:
        print("build_map: profile has no class records; import-time edits will run everything", file=sys.stderr)

    # ------------------------------------------------------------- default uses
    # PROFILE_DEFAULTS=1 runs note every call that left out an argument with a literal default.
    # An edit to such a default reaches those tests only: (rel, line, c0, c1) -> (bits or GLOBAL,
    # queries of the per-query test or None when not known per query).
    default_use = {}
    if args.defaults:
        dmeta = {}
        dcontexts = read_profile(args.defaults, dmeta)
        site_of = dmeta.get("default_sites", {})
        used_bits, used_queries = {}, {}
        for (path, site), (rel, line, c0, c1, _param, _def) in site_of.items():
            used_bits.setdefault((rel, line, c0, c1), 0)
            used_queries.setdefault((rel, line, c0, c1), set())
        for ctx, entry in dcontexts.items():
            found = entry.get("defaults")
            if not found:
                continue
            charged = tests.resolve(ctx)
            query = int(ctx[len(pq_prefix):]) if ctx.startswith(pq_prefix) else None
            for path, site in found:
                key = tuple(site_of[(path, site)][:4])
                if charged == GLOBAL or used_bits[key] == GLOBAL:
                    used_bits[key] = GLOBAL
                    continue
                used_bits[key] |= charged
                if charged & pq_bit:
                    used_queries[key].add(OUTSIDE if query is None else query)
        for key, bits in used_bits.items():
            queries = used_queries[key]
            default_use[key] = (bits, None if (OUTSIDE in queries or not bits & pq_bit) else tuple(sorted(queries)))
        print("build_map: %d literal defaults profiled, %d used by some test" % (
            len(default_use), sum(1 for b, _ in default_use.values() if b)), file=sys.stderr)

    # ------------------------------------------------------------- users of classes and modules
    # Edits nothing above models (an arg_types key renamed, a statement dropped, a module that
    # no longer imports) reach only tests that use the class or module they sit in: tests that
    # instantiated it or a subclass, generated SQL for it, read its attributes or ran its code.
    user_bits = defaultdict(int)
    user_global = set()
    for ctx, entry in contexts.items():
        found = entry.get("users")
        if not found:
            continue
        charged = tests.resolve(ctx)
        for key in found:
            if charged == GLOBAL:
                user_global.add(key)
            else:
                user_bits[key] |= charged
    owner_reads = defaultdict(int)
    for tag, bits in attr_reads.items():
        owner_reads[tag.rpartition(":")[0]] |= bits
    owner_global = {tag.rpartition(":")[0] for tag in attr_global}
    method_cov = defaultdict(int)
    module_cov = defaultdict(int)
    for rel, ix in files.items():
        if not rel.startswith("sqlglot/"):
            continue
        mod = module_name(rel)
        for ln, qual in ix.get("class_of", {}).items():
            bits = cov.get((rel, ln), 0)
            if bits:
                method_cov["%s.%s" % (mod, qual)] |= bits
    for (rel, ln), bits in cov.items():
        if rel.startswith("sqlglot/"):
            module_cov[module_name(rel)] |= bits
    templated = import_time_instances() if user_bits else set()
    class_cache = {}

    def class_users(key):
        if not user_bits:
            return GLOBAL
        if key not in class_cache:
            members = effects.family(key) if key in meta["families"] else {key}
            bits = 0
            for member in members:
                if member in templated or member in user_global or member in owner_global:
                    bits = GLOBAL
                    break
                bits |= user_bits.get(member, 0) | owner_reads.get(member, 0) | method_cov.get(member, 0)
            class_cache[key] = bits
        return class_cache[key]

    out_files = {}
    for rel, ix in files.items():
        kind = ix["kind"]
        spans = ix["spans"]
        sets = {}
        qsets = {}
        xq = {}
        checks = []
        def span_set(a, b):
            """The set id for an edit anywhere in lines a..b, with the per-query test's queries
            it reaches (None: unknown, so the whole test) and those it reaches after planning."""
            bits = union_eff(rel, range(a, b + 1))
            staged = set()
            for l2 in range(a, b + 1):
                staged |= stage_front.get((rel, l2), set()) | stage_back.get((rel, l2), set())
            if staged and bits != GLOBAL:
                bits |= pq_bit  # the per-query replay saw the test run it even if the profile did not
            sid = table.add(bits)
            if sid < 0 or not table.sets[sid] & pq_bit:
                return sid, None, ()
            queries = set()
            imported = False
            for l2 in range(a, b + 1):
                seen = qcov.get((rel, l2), set())
                if "unsplit" in seen and not any(isinstance(q, int) and q >= 0 for q in seen):
                    queries.add(OUTSIDE)  # only the unsplit runs saw the test here
                queries |= {q for q in seen if q != "unsplit"}
                if (rel, l2) in via_import:
                    imported = True
            queries |= staged
            span_set.runtime = None
            if imported and queries and OUTSIDE not in queries:
                # Runs at import too: per query only if what the import computed with it comes
                # out the same, which the selector checks (the values are listed in "imp").
                values = import_values(rel, a, b)
                if values is not None:
                    span_set.runtime = (tuple(sorted(queries)), values)
            if imported:
                queries.add(OUTSIDE)
            if not queries or OUTSIDE in queries:
                if span_set.runtime is not None:
                    runtime_queries = set(span_set.runtime[0])
                    back = set()
                    for l2 in range(a, b + 1):
                        back |= stage_back.get((rel, l2), set())
                        for q in qcov.get((rel, l2), ()):
                            if isinstance(q, int) and q >= 0 and q not in stage_front.get((rel, l2), ()) \
                                    and q not in stage_back.get((rel, l2), ()):
                                back.add(q)
                    span_set.runtime = (span_set.runtime[0], tuple(sorted(back & runtime_queries)), span_set.runtime[1])
                return sid, None, ()
            back = set()
            if stage_front or stage_back:
                for l2 in range(a, b + 1):
                    back |= stage_back.get((rel, l2), set())
                    for q in qcov.get((rel, l2), ()):
                        # seen by the profile but by neither stage of the replay: back
                        if isinstance(q, int) and q >= 0 and q not in stage_front.get((rel, l2), ()) \
                                and q not in stage_back.get((rel, l2), ()):
                            back.add(q)
            return sid, tuple(sorted(queries)), tuple(sorted(back & queries))

        def span_steps(a, b):
            mask = 0
            for l2 in range(a, b + 1):
                mask |= stage_modes.get((rel, l2), 0)
            return mask

        stg = {}
        qrt = {}  # line -> (queries, back queries, import-time values to compare), see span_set
        for ln in range(1, ix["n"] + 1):
            k = kind[ln - 1]
            if k not in (RUNTIME, MIXED):
                continue
            a, b = spans[ln - 1]
            sid, queries, back = span_set(a, b)
            if sid != NO_TESTS:
                sets[ln] = sid
                if queries:
                    qsets[ln] = queries
                    if back:
                        xq[ln] = back
                    steps = span_steps(a, b)
                    if steps:
                        stg[ln] = steps
                elif getattr(span_set, "runtime", None) is not None:
                    qrt[ln] = span_set.runtime
                    steps = span_steps(a, b)
                    if steps:
                        stg[ln] = steps
            if any((rel, l2) in via_import for l2 in range(a, b + 1)):
                checks.append(ln)  # the edited code also ran at import time; it could now fail there

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
        # entries of a module-level dict spliced into arg_types, whose other uses look at keys only
        if rel.startswith("sqlglot/"):
            for ln, entries in ix.get("dict_entries", {}).items():
                for qualname, attr, source, c0, c1 in entries:
                    origin = "dict:%s.%s" % (module_name(rel), attr)
                    if qualname or origin not in edges or not spliced_ok.get(origin):
                        continue
                    try:
                        arg = ast.literal_eval(source)
                    except (ValueError, SyntaxError):
                        continue
                    if not isinstance(arg, str):
                        continue
                    bits, glob = 0, False
                    for cls in descendants(origin):
                        if cls == origin:
                            continue
                        if cls in opaque:
                            glob = True
                        m = missing.get((cls, arg), 0)
                        if m == GLOBAL:
                            glob = True
                        else:
                            bits |= m
                    argt[ln] = GLOBAL if glob else table.add(bits)

        cattr = {}
        kattr = {}
        dflt = {}
        if rel.startswith("sqlglot/") and key_tags:
            for ln, entries in ix.get("dict_entries", {}).items():
                out = []
                for qualname, attr, source, c0, c1 in entries:
                    if not qualname:
                        continue  # module-level dicts are not profiled by key
                    k = effects.resolve_key(rel, qualname, source)
                    if k is None:
                        continue
                    owner = "%s.%s" % (module_name(rel), qualname)
                    out.append((c0, c1, table.add(effects.tests_for_key(owner, attr, k, keyed, displays))))
                if out:
                    kattr[ln] = out
        if rel.startswith("sqlglot/") and meta["families"]:
            pure = set(ix["pure"])
            for ln, entries in ix["classattr"].items():
                bits = 0
                check = ln not in pure
                for qualname, name in entries:
                    owner = "%s.%s" % (module_name(rel), qualname) if qualname else MODULE
                    got = effects.tests_for(owner, name)
                    if got == GLOBAL:
                        bits = GLOBAL
                        break
                    bits |= got
                    # import-time code that derives from the value could fail on the new value
                    if len(effects.affected(owner, name)) > 1:
                        check = True
                cattr[ln] = (table.add(bits), check)
        for ln, entries in ix["defaults"].items():
            # a default reaches whoever ran the function's body, or, when the default-use
            # profile covers it, whoever called the function without that argument:
            # (c0, c1, set, check, queries, back, steps)
            out = []
            for c0, c1, b0, b1 in entries:
                sid, queries, back = span_set(b0, b1)
                check = any((rel, l2) in via_import for l2 in range(b0, b1 + 1))
                used = default_use.get((rel, ln, c0, c1))
                if used is not None and used[0] != GLOBAL and sid != GLOBAL:
                    bits, used_queries = used
                    sid = table.add(bits)
                    if bits & pq_bit:
                        queries = used_queries
                        back = tuple(q for q in back if queries is not None and q in queries)
                    else:
                        queries, back = None, ()
                out.append((c0, c1, sid, check, queries, back, span_steps(b0, b1)))
            dflt[ln] = out

        class_fb = []
        if rel.startswith("sqlglot/"):
            spans = {}
            for ln, qual in ix.get("class_of", {}).items():
                a, b = spans.get(qual, (ln, ln))
                spans[qual] = (min(a, ln), max(b, ln))
            for qual, (a, b) in sorted(spans.items(), key=lambda x: x[1][1] - x[1][0]):
                class_fb.append((a, b, table.add(class_users("%s.%s" % (module_name(rel), qual)))))

        users = module_users.get(rel, 0)
        out_files[rel] = {
            "kind": kind,
            "flags": ix["flags"],
            "sets": sets,
            "cols": ix["runtime_cols"],
            "argt": argt,
            "cattr": cattr,
            "class_fb": class_fb,
            "kattr": kattr,
            "dflt": dflt,
            "checks": set(checks),
            "qsets": qsets,
            "xq": xq,
            "stg": stg,
            "qrt": qrt,
            "units": [x for unit in ix["units"] for x in unit],
            "annot": ix["annotations"],
            "pure": set(ix["pure"]),
            "users": GLOBAL if users == GLOBAL else table.add(users),
            "sha1": digests[rel],
        }

    openers = defaultdict(int)
    # A SQL fixture a test class's setUpClass loads into a list is read, for the profile, by
    # every test in the class. The tests that actually used that list are known (D records), so
    # for such a file there is also a narrower answer: (loader, its readers outside those class
    # fixtures plus the list's users, the same without the per-query test). The selector uses it
    # when the patched file still loads.
    provenance = meta.get("provenance", {})
    tags_of = defaultdict(set)
    for tag, (path, loader) in provenance.items():
        tags_of[path].add((tag, loader))
    narrow = defaultdict(int)
    data_users = defaultdict(int)
    for ctx, entry in contexts.items():
        found = entry.get("data")
        if found:
            charged = tests.resolve(ctx)
            for tag in found:
                data_users[tag] = GLOBAL if charged == GLOBAL or data_users[tag] == GLOBAL else data_users[tag] | charged
    for ctx, entry in contexts.items():
        charged = tests.resolve(ctx)
        for path in entry["reads"]:
            if "__pycache__" in path or (ctx == "<other>" and path.endswith("/")):
                continue  # bytecode caches, and pytest walking directories while collecting
            if charged == GLOBAL or openers[path] == GLOBAL:
                openers[path] = GLOBAL
            else:
                openers[path] |= charged
            class_fixture = ctx.startswith("<fixture>|class|") and path in tags_of
            if not class_fixture:
                narrow[path] = GLOBAL if charged == GLOBAL or narrow[path] == GLOBAL else narrow[path] | charged
    class_data = {}
    for path, entries in tags_of.items():
        loaders = {loader for _tag, loader in entries}
        bits = narrow.get(path, 0)
        for tag, _loader in entries:
            got = data_users.get(tag, 0)
            bits = GLOBAL if bits == GLOBAL or got == GLOBAL else bits | got
        if len(loaders) != 1 or bits == GLOBAL or openers.get(path) == GLOBAL:
            continue
        rel_fixture = path
        try:
            with open(os.path.join(src, rel_fixture), "rb") as f:
                digest = hashlib.sha1(f.read()).hexdigest()
        except OSError:
            continue
        class_data[path] = (loaders.pop(), table.add(bits), table.add(bits & ~pq_bit) if pq_bit else table.add(bits), digest)

    # ------------------------------------------------------------- weak mutation sites
    # Each weak run numbers its sites itself, so runs are merged by what a site is (file,
    # line, columns, text) rather than by its number. Every run covers the whole suite.
    weak_sites = 0
    if args.weak:
        weak_ran, weak_failed = read_outcomes(args.weak_outcomes) if args.weak_outcomes else ([], [])
        unprofiled = 0
        trusted = set(weak_ran) - set(weak_failed)
        for t, i in tests.index.items():
            if t not in trusted:
                unprofiled |= 1 << i
        if unprofiled:
            print("build_map: %d tests have no weak profile; they keep plain coverage" % bin(unprofiled).count("1"), file=sys.stderr)
        infected = defaultdict(int)  # (rel, ln, c0, c1, text, variant) -> bits or GLOBAL
        qinfected = defaultdict(set)  # same key -> queries of the per-query test (or OUTSIDE)
        known = set()
        weak_split = _split_runs(args.weak)
        for path in args.weak:
            wmeta = {"split_runs": weak_split}
            wcontexts = read_profile([path], wmeta)
            sites = wmeta.get("sites", [])
            for rel, ln, c0, c1, text, variants in sites:
                for vtext in variants:
                    known.add((rel, ln, c0, c1, text, vtext))
            for ctx, entry in wcontexts.items():
                codes = entry.get("weak")
                if not codes:
                    continue
                charged = tests.resolve(ctx)
                query = int(ctx[len(pq_prefix):]) if ctx.startswith(pq_prefix) else None
                if ctx.endswith("|call#unsplit"):
                    query = "unsplit"
                for code in codes:
                    site, index = divmod(code, 4)
                    rel, ln, c0, c1, text, variants = sites[site]
                    key = (rel, ln, c0, c1, text, variants[index])
                    if charged == GLOBAL or infected[key] == GLOBAL:
                        infected[key] = GLOBAL
                    else:
                        infected[key] |= charged
                        if charged & pq_bit:
                            qinfected[key].add(OUTSIDE if query is None else query)
        by_site = defaultdict(list)
        from weak_rewrite import impure_bool_tokens

        impure = {}
        for key in sorted(known):
            rel, ln, c0, c1, text, vtext = key
            if c0 < 0 or rel not in out_files:
                continue
            if text in ("and", "or"):
                if rel not in impure:
                    with open(os.path.join(src, rel), encoding="utf-8") as f:
                        impure[rel] = impure_bool_tokens(f.read(), rel)
                if (ln, c0) in impure[rel]:
                    continue  # a swap could skip a side effect: plain coverage instead
            bits = infected.get(key, 0)
            if bits == GLOBAL:
                continue  # it would differ while sqlglot is being imported: no shortcut
            fallback = cov.get((rel, ln), 0) & unprofiled
            sid = table.add(bits | fallback)
            queries = qinfected.get(key, set())
            if "unsplit" in queries:
                queries = {OUTSIDE} if not any(isinstance(x, int) and x >= 0 for x in queries) else queries - {"unsplit"}
            q = tuple(sorted(queries)) if queries and OUTSIDE not in queries and not fallback & pq_bit else None
            by_site[(rel, ln, c0, c1, text)].append((vtext, sid, q))
        for (rel, ln, c0, c1, text), usable in by_site.items():
            shown = text if len(text) == c1 - c0 else text.strip()  # the exact source text at c0..c1
            out_files[rel].setdefault("weak", {}).setdefault(ln, []).append((c0, c1, shown, usable))
            weak_sites += 1
    for info in out_files.values():
        info.setdefault("weak", {})

    scopes = test_module_scopes(src, tests)

    # Which sqlglot modules import each module while being imported, so the selector's import
    # check only has to load the patched modules and their importers.
    modules = {module_name(rel): rel for rel in files if rel.startswith("sqlglot/")}
    importers = defaultdict(set)
    for rel in modules.values():
        me = module_name(rel)
        package = me if rel.endswith("__init__.py") else me.rpartition(".")[0]
        for level, mod, names in files[rel]["imports"]:
            if level:
                base = package
                for _ in range(level - 1):
                    base = base.rpartition(".")[0]
                mod = base + ("." + mod if mod else "")
            targets = [mod] + [mod + "." + n for n in names]
            for target in targets:
                while target:
                    if target in modules and target != me:
                        importers[target].add(me)
                    target = target.rpartition(".")[0]
    for rel in modules.values():
        seen, stack = set(), [module_name(rel)]
        while stack:
            for imp in importers[stack.pop()]:
                if imp not in seen:
                    seen.add(imp)
                    stack.append(imp)
        out_files[rel]["importers"] = sorted(seen)

    core = set(core_modules(src))
    imports_by_test = test_module_imports(src, files, set(modules))
    # importlib.import_module calls per test (PROFILE_DATA=1), the dialect registry being reset
    # before each test so that loading a dialect by name always shows
    dyn_users = defaultdict(int)
    for ctx, entry in contexts.items():
        found = entry.get("dyn")
        if not found:
            continue
        charged = tests.resolve(ctx)
        for mod in found:
            dyn_users[mod] = GLOBAL if charged == GLOBAL or dyn_users[mod] == GLOBAL else dyn_users[mod] | charged
    # import statements inside sqlglot functions: (rel, line, modules the statement may import)
    try:
        import sqlglot.dialects as dialects_pkg

        by_attribute = dict(getattr(dialects_pkg, "MODULE_BY_ATTRIBUTE", {}))
    except Exception:  # noqa: BLE001
        by_attribute = {}
    lazy = []
    for rel, ix in files.items():
        if not rel.startswith("sqlglot/"):
            continue
        here = module_name(rel)
        package = here if rel.endswith("__init__.py") else here.rpartition(".")[0]
        for ln, level, mod, names in ix.get("lazy_imports", ()):
            base = mod
            if level:
                parts = package.split(".")
                base = ".".join(parts[: len(parts) - (level - 1)] + ([mod] if mod else []))
            targets = {base}
            for n in names:
                targets.add("%s.%s" % (base, n))
                if base == "sqlglot.dialects":
                    targets.add("sqlglot.dialects.%s" % by_attribute.get(n, n.lower()))
            lazy.append((rel, ln, {t for t in targets if t in modules}))
    for rel in modules.values():
        me = module_name(rel)
        group = {me} | set(out_files[rel]["importers"])
        if not user_bits or group & core:
            out_files[rel]["module_fb"] = GLOBAL
            continue
        bits = 0
        for mod in group:
            mod_rel = modules.get(mod)
            bits |= module_cov.get(mod, 0)
            for qual in set(files[mod_rel].get("class_of", {}).values()) if mod_rel else ():
                got = class_users("%s.%s" % (mod, qual))
                if got == GLOBAL:
                    bits = GLOBAL
                    break
                bits |= got
            if bits == GLOBAL:
                break
        if bits != GLOBAL:
            for test_rel, mods in imports_by_test.items():
                if mods & group:
                    bits |= tests.scope(test_rel)
            for mod in group:
                got = dyn_users.get(mod, 0)
                if got == GLOBAL:
                    bits = GLOBAL
                    break
                bits |= got
            for rel2, ln, targets in lazy:
                if targets & group:
                    got = union_eff(rel2, [ln])  # whoever ran that import statement
                    if got == GLOBAL:
                        bits = GLOBAL
                        break
                    bits |= got
        out_files[rel]["module_fb"] = table.add(bits) if bits != GLOBAL else GLOBAL

    blob = {
        "format": FORMAT,
        "commit": PINNED,
        "tests": tests.ids,
        "files": out_files,
        "reads": {p: (GLOBAL if b == GLOBAL else table.add(b)) for p, b in openers.items()},
        "class_data": class_data,
        "scopes": {p: table.add(b) for p, b in scopes.items()},
        # tests that import sqlglot in a fresh interpreter, which the profile cannot see into,
        # and the modules such an "import sqlglot" loads
        "fresh": [tests.index[t] for t in FRESH_INTERPRETER_TESTS if t in tests.index],
        "core": sorted(core),
        "per_query": per_query_reference(args.tpcds, pq_index, bool(args.tpcds_stages)),
        "sets": table.sets,
    }
    data = zlib.compress(marshal.dumps(blob), 9)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(data)
    print(
        "build_map: %d tests, %d files, %d sets, %d read paths, %d weak sites -> %s (%.1f MB)"
        % (len(tests.ids), len(out_files), len(table.sets), len(openers), weak_sites, args.out, len(data) / 1e6),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
