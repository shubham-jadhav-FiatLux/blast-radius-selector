"""Re-checks a few TPC-DS queries of tests/test_executor.py::TestExecutor::test_execute_tpcds.

That test executes every TPC-DS fixture query with sqlglot's Python executor (through a process
pool) and compares each result with duckdb's answer for ``transpile(sql, write="duckdb")``. At
the pinned commit it passes. For a given query, if a patched sqlglot still produces exactly the
same result table and exactly the same duckdb SQL, the comparison for that query cannot change.

The profile splits the test into one context per query, so the selector knows which queries an
edit can reach. When those are few, it runs them here (in a child process with the patched
sources on the import path) instead of paying for the whole 21-second test.

``fingerprints(root, queries)`` returns {query: (rows digest, duckdb digest)} using the same
steps as the test: the same table loading, the pool's pickle round trips, find_tables and
execute. Any exception propagates: the caller then keeps the test.
"""

import ast
import csv
import gzip
import hashlib
import os
import pickle
import sys
import time
import unittest  # noqa: F401  sqlglot's strict argument checks are on under tests, so they are here
import zlib

FIXTURE = "optimizer/tpc-ds/tpc-ds.sql"
_LOADED = {}  # the tables, loaded once per process


def _open(path):
    # tests/test_executor.py:open_file
    with open(path, "rb") as f:
        gzipped = f.read(2) == b"\x1f\x8b"
    if gzipped:
        return gzip.open(path, "rt", newline="")
    return open(path, encoding="utf-8", newline="")


def _tables(fixtures, schema):
    # TestExecutor.setUpClass, without the duckdb half
    from sqlglot.executor.table import Table

    directory = os.path.join(fixtures, "optimizer", "tpc-ds") + "/"
    tables = {}
    for table, columns in schema.items():
        reader = csv.reader(_open(f"{directory}{table}.csv.gz"), delimiter="|")
        rows = []
        ctypes = []
        next(reader)
        for row in reader:
            if not ctypes:
                for v in row:
                    try:
                        ctypes.append(type(ast.literal_eval(v)))
                    except (ValueError, SyntaxError):
                        ctypes.append(str)
            rows.append(tuple(None if (t is not str and v == "") else t(v) for t, v in zip(ctypes, row)))
        tables[table] = Table(columns=columns, rows=rows)
    return tables


def _digest(value):
    return hashlib.sha1(repr(value).encode("utf-8")).hexdigest()


def plan_digest(value):
    """A digest of everything the executor reads from a plan: its steps and how they depend on
    each other, and every expression in them with its arguments, type, comments and meta. Sets
    are digested in sorted order, so object identities never matter. Works on any value."""
    from sqlglot.expressions.core import Expr
    from sqlglot.planner import Step

    memo = {}
    atoms = (str, int, float, bool, bytes, type(None))

    def node(parts):
        return hashlib.blake2b(repr(parts).encode("utf-8"), digest_size=16).hexdigest()

    def fp(v):
        if isinstance(v, atoms):
            return "%s:%r" % (type(v).__name__, v)
        got = memo.get(id(v))
        if got is not None:
            return got
        if isinstance(v, Expr):
            parts = ["E", type(v).__module__, type(v).__qualname__]
            parts.extend((name, fp(arg)) for name, arg in v.args.items())
            parts.append(fp(getattr(v, "_type", None)))
            parts.append(fp(getattr(v, "comments", None)))
            parts.append(fp(getattr(v, "_meta", None)))
            out = node(parts)
        elif isinstance(v, Step):
            deps = sorted(fp(d) for d in v.dependencies)
            attrs = [(n, fp(a)) for n, a in sorted(vars(v).items()) if n not in ("dependencies", "dependents")]
            out = node(["S", type(v).__qualname__, deps, attrs])
        elif isinstance(v, (list, tuple)):
            out = node([type(v).__name__] + [fp(x) for x in v])
        elif isinstance(v, dict):
            out = node(["dict"] + [(fp(a), fp(b)) for a, b in v.items()])
        elif isinstance(v, (set, frozenset)):
            out = node([type(v).__name__] + sorted(fp(x) for x in v))
        elif isinstance(v, type):
            out = "T:%s.%s" % (v.__module__, v.__qualname__)
        elif hasattr(v, "__dict__"):
            out = node(["O", type(v).__module__, type(v).__qualname__, fp(vars(v))])
        else:
            out = node(["R", type(v).__qualname__, repr(v)])  # identities may leak in: a miss, not a match
        memo[id(v)] = out
        return out

    return fp(value)


def value_digest(value, memo=None):
    """A digest of a value computed while sqlglot was imported (a parsed template, a table of
    token types, a dict of builder functions...), comparable between two imports of the same
    modules: functions by their code, constants, defaults and closures; classes and modules by
    name; expressions as in plan_digest. Cycles are cut, identities never used."""
    import enum
    import re as _re
    import types

    from sqlglot.expressions.core import Expr

    if memo is None:
        memo = {}  # by id: a caller may share one across values that stay alive meanwhile
    atoms = (str, int, float, bool, bytes, complex, type(None))

    def node(parts):
        return hashlib.blake2b(repr(parts).encode("utf-8"), digest_size=16).hexdigest()

    def code(c):
        return node(["C", c.co_name, c.co_code, [fp(k) for k in c.co_consts], c.co_names, c.co_varnames,
                     c.co_freevars, c.co_cellvars, c.co_argcount, c.co_kwonlyargcount, c.co_flags])

    def fp(v):
        if isinstance(v, atoms):
            return "%s:%r" % (type(v).__name__, v)
        key = id(v)
        got = memo.get(key)
        if got is not None:
            return got
        memo[key] = "cycle"
        if isinstance(v, Expr):
            parts = ["E", type(v).__module__, type(v).__qualname__]
            parts.extend((n, fp(a)) for n, a in v.args.items())
            parts += [fp(getattr(v, "_type", None)), fp(getattr(v, "comments", None)), fp(getattr(v, "_meta", None))]
            out = node(parts)
        elif isinstance(v, enum.Enum):
            out = "N:%s.%s.%s" % (type(v).__module__, type(v).__qualname__, v.name)
        elif isinstance(v, type):
            out = "T:%s.%s" % (v.__module__, v.__qualname__)
        elif isinstance(v, types.ModuleType):
            out = "M:" + v.__name__
        elif isinstance(v, types.CodeType):
            out = code(v)
        elif isinstance(v, types.FunctionType):
            cells = []
            for cell in v.__closure__ or ():
                try:
                    cells.append(fp(cell.cell_contents))
                except ValueError:
                    cells.append("empty")
            out = node(["F", v.__module__, v.__qualname__, code(v.__code__), fp(v.__defaults__),
                        fp(v.__kwdefaults__), cells])
        elif isinstance(v, (types.MethodType,)):
            out = node(["B", fp(v.__func__), fp(type(v.__self__))])
        elif isinstance(v, (staticmethod, classmethod)):
            out = node([type(v).__name__, fp(v.__func__)])
        elif isinstance(v, property):
            out = node(["P", fp(v.fget), fp(v.fset), fp(v.fdel)])
        elif isinstance(v, (types.BuiltinFunctionType, types.BuiltinMethodType)):
            out = "X:%s" % getattr(v, "__qualname__", repr(v))
        elif isinstance(v, _re.Pattern):
            out = node(["re", v.pattern, v.flags])
        elif isinstance(v, (list, tuple)) and v and all(isinstance(x, type) for x in v):
            # a sequence of classes is in practice a set turned into a tuple for isinstance()
            # (_CANONICALIZE_TYPES): its order comes from set iteration, new on every import
            out = node([type(v).__name__ + "-of-classes"] + sorted(fp(x) for x in v))
        elif isinstance(v, (list, tuple)):
            out = node([type(v).__name__] + [fp(x) for x in v])
        elif isinstance(v, dict):
            # as a mapping: tables built by iterating sets of classes come out in an order that
            # differs from one import to the next already, so nothing can rely on it
            out = node(["dict"] + sorted((fp(a), fp(b)) for a, b in v.items()))
        elif isinstance(v, (set, frozenset)):
            out = node([type(v).__name__] + sorted(fp(x) for x in v))
        else:
            fields = []
            if hasattr(v, "__dict__"):
                fields.append(fp(vars(v)))
            for klass in type(v).__mro__:
                for slot in getattr(klass, "__slots__", ()) or ():
                    if isinstance(slot, str) and hasattr(v, slot):
                        fields.append((slot, fp(getattr(v, slot))))
            # (a repr may name the object's address: a lock made by each import is a new one)
            out = node(["O", type(v).__module__, type(v).__qualname__, fields]) if fields else \
                node(["R", type(v).__qualname__, _re.sub(r" at 0x[0-9a-fA-F]+", "", repr(v))])
        memo[key] = out
        return out

    return fp(value)


def import_value(spec):
    """The value of (module, class qualname or "", name) as now imported."""
    import importlib

    module, qualname, name = spec
    obj = importlib.import_module(module)
    for part in qualname.split(".") if qualname else ():
        obj = getattr(obj, part)
    return vars(obj)[name] if isinstance(obj, type) and name in vars(obj) else getattr(obj, name)


class _Planned(Exception):
    pass


def encode_tree(root):
    """An exact, picklable copy of an expression tree: every argument as it was set (sqlglot's
    own pickling drops arguments that are None or empty lists, and the planner tells an empty
    list from a missing one), shared nodes still shared, types, comments and meta."""
    from sqlglot.expressions.core import Expr

    nodes, index, stack = [], {}, []

    def ref(v):
        if isinstance(v, Expr):
            k = id(v)
            if k not in index:
                index[k] = len(nodes)
                nodes.append(None)
                stack.append(v)
            return ("E", index[k])
        if type(v) is list:
            return ("L", [ref(x) for x in v])
        return ("V", v)

    top = ref(root)
    while stack:
        node = stack.pop()
        nodes[index[id(node)]] = (
            type(node).__module__, type(node).__qualname__,
            [(k, ref(v)) for k, v in node.args.items()],
            None if node._type is None else ref(node._type),
            None if node.comments is None else list(node.comments),
            None if node._meta is None else [(k, ref(v)) for k, v in node._meta.items()],
        )
    return zlib.compress(pickle.dumps((top, nodes), protocol=pickle.HIGHEST_PROTOCOL), 6)


def decode_tree(blob):
    """The tree encode_tree saved, built from the expression classes imported now."""
    import importlib

    top, nodes = pickle.loads(zlib.decompress(blob))
    objs = []
    for mod, qualname, *_ in nodes:
        cls = sys.modules.get(mod) or importlib.import_module(mod)
        for part in qualname.split("."):
            cls = getattr(cls, part)
        objs.append(object.__new__(cls))

    def val(r):
        kind, x = r
        if kind == "E":
            return objs[x]
        if kind == "L":
            return [val(y) for y in x]
        return x

    for node, (_mod, _q, args, typ, comments, meta) in zip(objs, nodes):
        node.args = {k: val(r) for k, r in args}
        node.parent = node.arg_key = node.index = None
        node.comments = comments
        node._type = None if typ is None else val(typ)
        node._meta = None if meta is None else {k: val(r) for k, r in meta}
        node._hash = None
    for node in objs:
        if not node.is_primitive:
            for k, v in node.args.items():
                node._set_parent(k, v)
    return val(top)


def front_digests(root, queries, seconds=None):
    """{query: (plan digest, duckdb digest)}; see iter_front_digests."""
    return dict(iter_front_digests(root, sorted(set(queries)), seconds))


def iter_front_digests(root, queries, seconds=None):
    """{query: (plan digest, duckdb digest)} from the steps of the test up to the plan, run as the
    test runs them (parse_one, the pickle round trip, find_tables, then sqlglot.executor.execute
    with the executor that would run the plan replaced by one that only takes the plan), plus
    transpile(sql, write="duckdb"). Queries the test does not execute only get parsed and
    pickled: (None, None). Tables are stood in for by empty ones with the same columns: nothing
    before the plan reads their rows."""
    if root not in sys.path:
        sys.path.insert(0, root)
    import sqlglot.executor as executor
    from sqlglot import find_tables, parse_one, transpile
    from sqlglot.executor.table import Table
    from tests.helpers import TPCDS_SCHEMA, load_sql_fixture_pairs

    sqls = list(load_sql_fixture_pairs(FIXTURE))
    schema = pickle.loads(pickle.dumps(TPCDS_SCHEMA))
    tables = {name: Table(columns=list(columns)) for name, columns in schema.items()}
    taken = {}

    class TakesPlan:
        def __init__(self, env=None, tables=None):
            taken["tables"] = sorted(tables.mapping) if hasattr(tables, "mapping") else repr(tables)

        def execute(self, plan):
            taken["plan"] = plan
            raise _Planned

    original = executor.PythonExecutor
    try:
        for k in queries:
            started = time.process_time()
            meta, sql, _ = sqls[k]
            expression = pickle.loads(pickle.dumps(parse_one(sql)))
            if not meta.get("execute"):
                if seconds is not None:
                    seconds[k] = time.process_time() - started
                yield k, (None, None)
                continue
            found = {t.name: tables[t.name] for t in find_tables(expression)}
            taken.clear()
            executor.PythonExecutor = TakesPlan
            try:
                executor.execute(expression, schema=schema, tables=found)
            except _Planned:
                pass
            finally:
                executor.PythonExecutor = original
            if "plan" not in taken:
                raise RuntimeError("query %d never reached the executor" % k)
            plan = _digest((sorted(found), taken["tables"], plan_digest(taken["plan"])))
            duck = _digest(transpile(sql, write="duckdb")[0])
            if seconds is not None:
                seconds[k] = time.process_time() - started
            yield k, (plan, duck)
    finally:
        executor.PythonExecutor = original


# Steps of a query's way to its plan, as the stage profile names them. An edit whose lines run
# only in steps from "rule0" to "plan" (and maybe "tables" or "duckdb") can be re-checked from
# checkpoints: the input each step had at the pinned commit, saved when the image was built.
STEPS = ["parse", "pickle", "tables", "pre"] + ["rule%d" % i for i in range(14)] + ["plan", "duckdb"]
STEP_BIT = {name: 1 << i for i, name in enumerate(STEPS)}
FIRST_RULE, PLAN, DUCKDB = STEPS.index("rule0"), STEPS.index("plan"), STEPS.index("duckdb")
REPLAYABLE = (STEP_BIT["parse"] | STEP_BIT["pickle"] | STEP_BIT["tables"] | STEP_BIT["duckdb"]
              | sum(STEP_BIT[n] for n in STEPS[FIRST_RULE:PLAN + 1]))
PARSED = STEP_BIT["parse"] | STEP_BIT["pickle"]
SCHEMA = "<schema>"


def _rules_default_index(optimize):
    import inspect

    params = [p for p in inspect.signature(optimize).parameters.values() if p.kind == p.POSITIONAL_OR_KEYWORD]
    return [p.name for p in params].index("rules") - (len(params) - len(optimize.__defaults__))


def capture_checkpoints(root, k):
    """Runs query k the way front_digests does, keeping what each step got and gave: the
    pickled input of every optimizer rule and of Plan, the keyword arguments optimize() passed
    each rule, digests of their outputs, the tables find_tables found, the plan's digest and
    the duckdb SQL's. Then replays every step from its pickled input and keeps only the
    checkpoints that reproduce the same output ("ok")."""
    import inspect

    import sqlglot.executor as executor
    import sqlglot.optimizer.optimizer as optimizer
    from sqlglot import find_tables, parse_one, transpile
    from sqlglot.executor.table import Table
    from sqlglot.schema import Schema
    from tests.helpers import TPCDS_SCHEMA, load_sql_fixture_pairs

    meta, sql, _ = list(load_sql_fixture_pairs(FIXTURE))[k]
    started = time.process_time()
    e1 = pickle.loads(pickle.dumps(parse_one(sql)))
    parse_seconds = time.process_time() - started
    if not meta.get("execute"):
        return None
    schema = pickle.loads(pickle.dumps(TPCDS_SCHEMA))
    tables = {name: Table(columns=list(columns)) for name, columns in schema.items()}
    found = {t.name: tables[t.name] for t in find_tables(e1)}
    rules = optimizer.RULES
    inputs, outs, kwargs_of = [None] * len(rules), [None] * len(rules), [None] * len(rules)
    spent = [0.0] * (len(rules) + 1)  # each rule's CPU seconds, then Plan's
    taken = {}

    def step(i, rule):
        def run(expression, **kwargs):
            inputs[i] = encode_tree(expression)
            kwargs_of[i] = {n: (SCHEMA if isinstance(v, Schema) else v) for n, v in kwargs.items()}
            started = time.process_time()
            result = rule(expression, **kwargs)
            spent[i] = time.process_time() - started
            outs[i] = plan_digest(result)
            return result

        run.__signature__ = inspect.signature(rule)
        return run

    class TakesPlan:
        def __init__(self, env=None, tables=None):
            pass

        def execute(self, plan):
            taken["plan"] = plan
            raise _Planned

    original_plan = executor.Plan

    def planned(expression):
        taken["plan_input"] = encode_tree(expression)
        started = time.process_time()
        plan = original_plan(expression)
        spent[-1] = time.process_time() - started
        return plan

    defaults = list(optimizer.optimize.__defaults__)
    at = _rules_default_index(optimizer.optimize)
    saved = (executor.PythonExecutor, executor.Plan, optimizer.optimize.__defaults__)
    defaults[at] = tuple(step(i, r) for i, r in enumerate(rules))
    executor.PythonExecutor, executor.Plan = TakesPlan, planned
    optimizer.optimize.__defaults__ = tuple(defaults)
    try:
        executor.execute(e1, schema=schema, tables=found)
    except _Planned:
        pass
    finally:
        executor.PythonExecutor, executor.Plan, optimizer.optimize.__defaults__ = saved
    if "plan" not in taken or any(x is None for x in inputs):
        return None
    point = {
        "e1": encode_tree(e1),
        "names": sorted(t.name for t in find_tables(e1)),
        "inputs": inputs, "kwargs": kwargs_of, "outs": outs,
        "plan_input": taken["plan_input"], "plan": plan_digest(taken["plan"]),
        "duckdb": _digest(transpile(sql, write="duckdb")[0]), "sql_k": k,
        "seconds": [round(x, 5) for x in spent] + [round(parse_seconds, 5)],
        # what parse_one and the pickle round trip give: everything after starts from it
        "e1_digest": plan_digest(e1),
    }
    again = plan_digest(pickle.loads(pickle.dumps(parse_one(sql))))
    point["parse_ok"] = again == point["e1_digest"]
    # each checkpoint must reproduce its step's output exactly when replayed on its own
    point["ok"] = [replay_steps(point, i, i) == outs[i] for i in range(len(rules))]
    point["plan_ok"] = replay_steps(point, PLAN - FIRST_RULE, PLAN - FIRST_RULE) == point["plan"]
    return point


def replay_steps(point, first, last):
    """Digest of what rules first..last (indices into RULES; len(RULES) stands for Plan) give
    when started from the checkpointed input of ``first``, with the sqlglot now imported."""
    import sqlglot.optimizer.optimizer as optimizer
    from sqlglot.planner import Plan
    from sqlglot.schema import ensure_schema
    from tests.helpers import TPCDS_SCHEMA

    rules = optimizer.RULES
    value = decode_tree(point["inputs"][first] if first < len(rules) else point["plan_input"])
    schema = None
    for i in range(first, last + 1):
        if i < len(rules):
            kwargs = dict(point["kwargs"][i])
            for n, v in kwargs.items():
                if v == SCHEMA:
                    if schema is None:
                        schema = ensure_schema(pickle.loads(pickle.dumps(TPCDS_SCHEMA)))
                    kwargs[n] = schema
            value = rules[i](value, **kwargs)
        else:
            value = Plan(value)
    return plan_digest(value)


def replay_cost(step_seconds, steps):
    """Seconds a checkpoint_check of these steps should take for one query, or None if only a
    full front pass can answer."""
    if step_seconds is None or not steps or steps & ~REPLAYABLE:
        return None
    span = [i for i in range(PLAN - FIRST_RULE + 1) if steps & (1 << (FIRST_RULE + i))]
    cost = 0.004  # decoding the checkpoint
    if span:
        cost += sum(step_seconds[min(span):max(span) + 1])
    if steps & PARSED:
        if len(step_seconds) <= PLAN - FIRST_RULE + 1:
            return None
        cost += step_seconds[PLAN - FIRST_RULE + 1]
    if steps & STEP_BIT["duckdb"]:
        cost += 0.006
    if steps & STEP_BIT["tables"]:
        cost += 0.002
    return cost


def checkpoint_check(root, point, steps):
    """True if query's steps in the ``steps`` bitmask, replayed from the checkpoints with the
    patched sources, give exactly what they gave at the pinned commit. None when the
    checkpoints cannot answer (a step outside the replayable ones, or one that did not
    reproduce itself when the image was built)."""
    if root not in sys.path:
        sys.path.insert(0, root)
    if steps & ~REPLAYABLE:
        return None
    if steps & PARSED:
        # parsing (and the pickle round trip into the worker) must give the very same tree;
        # then every later step starts from what it started from at the pinned commit
        if not point.get("parse_ok"):
            return None
        from sqlglot import parse_one
        from tests.helpers import load_sql_fixture_pairs

        sql = list(load_sql_fixture_pairs(FIXTURE))[point["sql_k"]][1]
        if plan_digest(pickle.loads(pickle.dumps(parse_one(sql)))) != point["e1_digest"]:
            return False
    nrules = len(point["inputs"])
    span = [i - FIRST_RULE for i in range(FIRST_RULE, PLAN + 1) if steps & (1 << i)]
    if span:
        first, last = min(span), max(span)
        if not (point["ok"][first] if first < nrules else point["plan_ok"]):
            return None
        want = point["outs"][last] if last < nrules else point["plan"]
        if replay_steps(point, first, last) != want:
            return False
    if steps & STEP_BIT["tables"]:
        from sqlglot import find_tables

        if sorted(t.name for t in find_tables(decode_tree(point["e1"]))) != point["names"]:
            return False
    if steps & STEP_BIT["duckdb"]:
        from sqlglot import transpile
        from tests.helpers import load_sql_fixture_pairs

        return None if point.get("sql_k") is None else (
            _digest(transpile(list(load_sql_fixture_pairs(FIXTURE))[point["sql_k"]][1], write="duckdb")[0])
            == point["duckdb"])
    return True


def fingerprints(root, queries):
    if root not in sys.path:
        sys.path.insert(0, root)
    from sqlglot import find_tables, parse_one, transpile
    from sqlglot.executor import execute
    from tests.helpers import FIXTURES_DIR, TPCDS_SCHEMA, load_sql_fixture_pairs

    sqls = list(load_sql_fixture_pairs(FIXTURE))
    wanted = sorted(set(queries))
    executed = [k for k in wanted if sqls[k][0].get("execute")]
    schema, tables = None, None
    if executed:
        # the pool initializer receives (schema, tables) through pickle
        if "tables" not in _LOADED:
            _LOADED["tables"] = pickle.loads(pickle.dumps((TPCDS_SCHEMA, _tables(FIXTURES_DIR, TPCDS_SCHEMA))))
        schema, tables = _LOADED["tables"]
    out = {}
    for k in wanted:
        meta, sql, _ = sqls[k]
        expression = pickle.loads(pickle.dumps(parse_one(sql)))
        if not meta.get("execute"):
            out[k] = (None, None)
            continue
        found = {t.name: tables[t.name] for t in find_tables(expression)}
        table = pickle.loads(pickle.dumps(execute(expression, schema=schema, tables=found)))
        rows = _digest((list(table.columns), [tuple(row) for row in table.rows]))
        out[k] = (rows, _digest(transpile(sql, write="duckdb")[0]))
    return out


def write_plans(root, target):
    """Writes the front digests of every query and what computing them costs, for the image:
    tpcds_check.py --plans <sources> <out.json>. Never fails the build: without the file the
    selector simply never compares plans."""
    import json
    import time

    try:
        sys.path.insert(0, root)
        from tests.helpers import load_sql_fixture_pairs

        count = len(list(load_sql_fixture_pairs(FIXTURE)))
        front_digests(root, [0])  # imports and first-use costs out of the way
        alone = {k: front_digests(root, [k])[k] for k in range(count)}
        timed = {}
        together = front_digests(root, range(count), timed)  # one pass, as a re-check runs them
        if any(alone[k] != together[k] for k in range(count)):
            raise RuntimeError("plan digests depend on which queries ran before")
        again = {}
        front_digests(root, range(count), again)  # the cheaper of two passes, against noise
        out = {"plans": [together[k][0] for k in range(count)],
               "duckdb": [together[k][1] for k in range(count)],
               "seconds": [round(min(timed[k], again[k]), 4) for k in range(count)]}
    except Exception as e:  # noqa: BLE001
        print("tpcds_check: no plan digests:", repr(e), file=sys.stderr)
        return
    try:
        points = {}
        for k in range(count):
            point = capture_checkpoints(root, k)
            if point is not None:
                points[k] = point
        bad = sum(1 for p in points.values() for ok in p["ok"] + [p["plan_ok"]] if not ok)
        with open(os.path.splitext(target)[0] + "_points.pickle", "wb") as f:
            pickle.dump(points, f, protocol=pickle.HIGHEST_PROTOCOL)
        out["step_seconds"] = [points[k]["seconds"] if k in points else None for k in range(count)]
        print("tpcds_check: checkpoints for %d queries, %d steps that do not replay" % (len(points), bad), file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print("tpcds_check: no checkpoints:", repr(e), file=sys.stderr)
    with open(target, "w") as f:
        json.dump(out, f)


_DUCK = {}


def duck_digest(root, sql, raw=False):
    """A digest of what duckdb returns for ``transpile(sql, write="duckdb")`` on the TPC-DS views,
    built as TestExecutor.setUpClass builds them. Equal for the patched and the pinned SQL means
    the test's comparison of the two sides is what it was, the executor's side being unchanged."""
    import site

    site.main()  # the selector runs with -S
    import duckdb

    from sqlglot import transpile

    conn = _DUCK.get("conn")
    if conn is None:
        if root not in sys.path:
            sys.path.insert(0, root)
        from tests.helpers import TPCDS_SCHEMA

        conn = duckdb.connect()
        directory = os.path.join(root, "tests", "fixtures", "optimizer", "tpc-ds") + "/"
        for table, columns in TPCDS_SCHEMA.items():
            conn.execute(
                f"""
                CREATE VIEW {table} AS
                SELECT *
                FROM READ_CSV('{directory}{table}.csv.gz', delim='|', header=True, columns={columns})
                """
            )
        _DUCK["conn"] = conn
    frame = conn.execute(transpile(sql, write="duckdb")[0]).fetchdf()
    if raw:
        return frame
    return _digest((list(frame.columns), [tuple(repr(v) for v in row) for row in frame.itertuples(index=False)]))


def text_passes(root, meta, sql):
    """test_execute_tpcds for one query whose text in the fixture the patch changed, as the test
    runs it: parsed (and pickled to the pool) like every query, and when its meta says execute,
    run by sqlglot's executor and compared with duckdb's answer for the same text the way
    subtestHelper compares them. Raises where the test would fail; True otherwise."""
    if root not in sys.path:
        sys.path.insert(0, root)
    from sqlglot import find_tables, parse_one
    from sqlglot.executor import execute
    from tests.helpers import FIXTURES_DIR, TPCDS_SCHEMA

    expression = pickle.loads(pickle.dumps(parse_one(sql)))
    if not meta.get("execute"):
        return True  # mp_execute returns None for it: nothing is compared
    if "tables" not in _LOADED:
        _LOADED["tables"] = pickle.loads(pickle.dumps((TPCDS_SCHEMA, _tables(FIXTURES_DIR, TPCDS_SCHEMA))))
    schema, tables = _LOADED["tables"]
    found = {t.name: tables[t.name] for t in find_tables(expression)}
    table = pickle.loads(pickle.dumps(execute(expression, schema=schema, tables=found)))
    a = duck_digest(root, sql, raw=True)
    import numpy as np
    import pandas as pd
    from pandas.testing import assert_frame_equal

    b = pd.DataFrame(((np.nan if c is None else c for c in r) for r in table.rows), columns=table.columns)
    assert_frame_equal(a, b, check_dtype=False, check_index_type=False)
    return True


def write_duck(root, target):
    """duckdb's answer digest for each executed query at the pinned commit (build time)."""
    import json

    sys.path.insert(0, root)
    from tests.helpers import load_sql_fixture_pairs

    out = {}
    try:
        for k, (meta, sql, _) in enumerate(load_sql_fixture_pairs(FIXTURE)):
            if meta.get("execute"):
                out[k] = duck_digest(root, sql)
        with open(target, "w") as f:
            json.dump(out, f)
    except Exception as e:  # noqa: BLE001
        print("tpcds_check: no duckdb answers:", repr(e), file=sys.stderr)


def main():
    """Writes the reference fingerprints and timings: tpcds_check.py <sources> <out.json>"""
    import json
    import time

    if sys.argv[1] == "--plans":
        write_plans(sys.argv[2], sys.argv[3])
        return
    if sys.argv[1] == "--duck":
        write_duck(sys.argv[2], sys.argv[3])
        return
    root, target = sys.argv[1], sys.argv[2]
    sys.path.insert(0, root)
    from tests.helpers import load_sql_fixture_pairs

    count = len(list(load_sql_fixture_pairs(FIXTURE)))
    start = time.process_time()
    fingerprints(root, [])
    reference = {"queries": [], "load_seconds": 0.0}
    t0 = time.process_time()
    from sqlglot.executor.table import Table  # noqa: F401
    from tests.helpers import FIXTURES_DIR, TPCDS_SCHEMA

    _tables(FIXTURES_DIR, TPCDS_SCHEMA)
    reference["load_seconds"] = round(time.process_time() - t0, 3)
    for k in range(count):
        t = time.process_time()
        rows, duck = fingerprints(root, [k])[k]
        spent = time.process_time() - t
        if rows is not None:
            spent -= reference["load_seconds"]  # each call above reloads the tables
        reference["queries"].append({"rows": rows, "duckdb": duck, "seconds": round(max(spent, 0.0), 3)})
    reference["total_seconds"] = round(time.process_time() - start, 1)
    with open(target, "w") as f:
        json.dump(reference, f, indent=0)


if __name__ == "__main__":
    main()
