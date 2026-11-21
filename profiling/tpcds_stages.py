"""Which lines of test_execute_tpcds run in which step of each query, before and after its plan.

For each TPC-DS query the test runs parse_one (in the main process), a pickle round trip into a
worker, find_tables, and sqlglot.executor.execute: that ensures the tables and the schema,
optimizes (one rule after another), plans, and finally runs PythonExecutor over the data. The
result table comes back through pickle and is compared with duckdb's answer for
transpile(sql, write="duckdb").

Everything up to the plan takes about 0.1 s a query; running the plan is most of the test's 21 s.
This records, per query, the lines that run in the steps up to the plan ("front") and the lines
that run anywhere else ("back"): the executor, ensure_tables, the body of execute() itself and
the result's pickle round trip. It also records, for every front line, in which steps it ran
("modes"): parse, pickle, tables (find_tables), pre (the rest of execute() and optimize()
around the rules), each optimizer rule, plan, duckdb. An edit only the front reaches can then be
re-checked by comparing plans, and an edit only some rules reach by re-running those rules alone.

Each query runs in a fresh fork of a process that has only imported sqlglot and loaded the
tables, twice (cold caches, then warm), and both runs count. Anything outside the windows this
script knows to be front counts as back. Nothing that runs while a module is being imported is
recorded: the main profile accounts for import-time effects.

    tpcds_stages.py <sqlglot sources> <out.json>
"""

import inspect
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FRONT_TOOL, BACK_TOOL = 3, 4
CO_OPTIMIZED = 0x1  # set on function code, unset on module and class bodies


def rules_default_index(optimize):
    """Where the ``rules`` default sits in optimize.__defaults__."""
    params = [p for p in inspect.signature(optimize).parameters.values() if p.kind == p.POSITIONAL_OR_KEYWORD]
    first = len(params) - len(optimize.__defaults__)
    return [p.name for p in params].index("rules") - first


def main():
    root, target = os.path.abspath(sys.argv[1]), sys.argv[2]
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.join(HERE, "..", "src"))
    import tpcds_check

    import sqlglot.executor as executor
    import sqlglot.optimizer.optimizer as optimizer
    from sqlglot import find_tables, parse_one, transpile
    from tests.helpers import FIXTURES_DIR, TPCDS_SCHEMA, load_sql_fixture_pairs

    sqls = list(load_sql_fixture_pairs(tpcds_check.FIXTURE))
    schema, tables = pickle.loads(pickle.dumps((TPCDS_SCHEMA, tpcds_check._tables(FIXTURES_DIR, TPCDS_SCHEMA))))
    execute_lines = sorted({ln for _, _, ln in executor.execute.__code__.co_lines() if ln})
    optimize_code = optimizer.optimize.__code__
    prefix = root + os.sep
    mon = sys.monitoring
    state = {"mode": None}
    front_seen = set()  # (file, line, mode)
    back_seen = set()  # (file, line)

    def importing():
        frame = sys._getframe(2)
        while frame is not None:
            g = frame.f_globals
            if frame.f_code.co_name == "<module>" and g.get("__spec__") is not None and g.get("__name__") != "__main__":
                return True
            frame = frame.f_back
        return False

    # Code that runs while a module is being imported (module and class bodies, and the
    # templates they parse) is the main profile's business: it charges such lines to the
    # readers of what the import computed. Not recorded, and kept armed.
    def on_front_line(code, line):
        if not code.co_flags & CO_OPTIMIZED or importing():
            return None
        mode = "pre" if code is optimize_code else state["mode"]
        front_seen.add((code.co_filename, line, mode))
        return None  # stays armed: the same line may run again in another step

    def on_back_line(code, line):
        if not code.co_flags & CO_OPTIMIZED or importing():
            return None
        back_seen.add((code.co_filename, line))
        return mon.DISABLE

    def set_mode(mode):
        previous = state["mode"]
        state["mode"] = mode
        mon.set_events(FRONT_TOOL, mon.events.LINE if mode not in (None, "back") else 0)
        mon.set_events(BACK_TOOL, mon.events.LINE if mode == "back" else 0)
        return previous

    def within(mode, call, *args, **kwargs):
        previous = set_mode(mode)
        try:
            return call(*args, **kwargs)
        finally:
            if state["mode"] == mode:
                set_mode(previous)

    original_ensure = executor.ensure_tables

    def ensure_tables(*args, **kwargs):
        previous = set_mode("back")
        try:
            return original_ensure(*args, **kwargs)
        finally:
            set_mode(previous)

    original_init = executor.PythonExecutor.__init__

    def executor_init(self, *args, **kwargs):
        set_mode("back")  # from here to the end of the query, everything is the back
        original_init(self, *args, **kwargs)

    original_plan = executor.Plan

    def plan(*args, **kwargs):
        return within("plan", original_plan, *args, **kwargs)

    def rule_step(index, rule):
        def step(*args, **kwargs):
            previous = set_mode("rule%d" % index)
            try:
                return rule(*args, **kwargs)
            finally:
                set_mode(previous)

        step.__signature__ = inspect.signature(rule)  # optimize() passes kwargs by signature
        return step

    defaults = list(optimizer.optimize.__defaults__)
    at = rules_default_index(optimizer.optimize)
    assert defaults[at] is optimizer.RULES
    defaults[at] = tuple(rule_step(i, r) for i, r in enumerate(optimizer.RULES))

    def run_query(meta, sql):
        expression = within("parse", parse_one, sql)
        expression = within("pickle", lambda e: pickle.loads(pickle.dumps(e)), expression)
        if not meta.get("execute"):
            return
        found = within("tables", lambda e: {t.name: tables[t.name] for t in find_tables(e)}, expression)
        set_mode("back")
        table = within("pre", executor.execute, expression, schema=schema, tables=found)
        set_mode("back")
        pickle.loads(pickle.dumps(table))
        within("duckdb", transpile, sql, write="duckdb")

    out = {"front": {}, "back": {}, "modes": {}, "executed": [], "rules": [r.__name__ for r in optimizer.RULES]}
    for k, (meta, sql, _) in enumerate(sqls):
        if meta.get("execute"):
            out["executed"].append(k)
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(read_fd)
            try:
                mon.use_tool_id(FRONT_TOOL, "tpcds-front")
                mon.register_callback(FRONT_TOOL, mon.events.LINE, on_front_line)
                mon.use_tool_id(BACK_TOOL, "tpcds-back")
                mon.register_callback(BACK_TOOL, mon.events.LINE, on_back_line)
                executor.ensure_tables = ensure_tables
                executor.PythonExecutor.__init__ = executor_init
                executor.Plan = plan
                optimizer.optimize.__defaults__ = tuple(defaults)
                for _ in range(2):  # cold, then warm
                    set_mode("back")
                    run_query(meta, sql)
                set_mode(None)
                result = {
                    "front": sorted({(f[len(prefix):], ln, m) for f, ln, m in front_seen if f.startswith(prefix)}),
                    "back": sorted({(f[len(prefix):], ln) for f, ln in back_seen if f.startswith(prefix)}),
                }
                payload = pickle.dumps(result)
                code = 0
            except BaseException as e:  # noqa: BLE001
                payload = pickle.dumps({"error": repr(e)})
                code = 1
            with os.fdopen(write_fd, "wb") as f:
                f.write(payload)
            os._exit(code)
        os.close(write_fd)
        with os.fdopen(read_fd, "rb") as f:
            result = pickle.loads(f.read())
        os.waitpid(pid, 0)
        if "error" in result:
            raise SystemExit("query %d failed: %s" % (k, result["error"]))
        for rel, ln, mode in result["front"]:
            out["front"].setdefault(rel, {}).setdefault(ln, set()).add(k)
            out["modes"].setdefault(rel, {}).setdefault(ln, set()).add(mode)
        for rel, ln in result["back"]:
            out["back"].setdefault(rel, {}).setdefault(ln, set()).add(k)
        if meta.get("execute"):
            for ln in execute_lines:
                out["back"].setdefault("sqlglot/executor/__init__.py", {}).setdefault(ln, set()).add(k)
        print("query", k, "front", len(result["front"]), "back", len(result["back"]), file=sys.stderr)
    for name in ("front", "back", "modes"):
        out[name] = {rel: {str(ln): sorted(v) for ln, v in lines.items()} for rel, lines in out[name].items()}
    with open(target, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
