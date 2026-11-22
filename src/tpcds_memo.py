"""Step records of tests/test_executor.py::TestExecutor::test_execute_tpcds at the pinned commit,
and a replay that runs only the executor steps an edit can reach.

The test runs each executed TPC-DS query through sqlglot's Python executor, which executes the
query's plan one step at a time (PythonExecutor._execute): a step reads the tables of the steps it
depends on and returns tables of its own. Most of the test's time goes there.

``record`` (build time) runs every executed query as the test does, with that loop replaced by a
copy that keeps, for each step, its output tables and the sqlglot functions that ran while it did
(sys.monitoring line events), and apart from those, the functions that ran around the steps: the
executor's set-up, the loop itself and the loading of the tables. A step is known by a digest of
its definition and of its dependencies' (tpcds_check.plan_digest), so a step of a plan that comes
out the same under a patch finds its record.

``Replay`` (check time, patched sources) runs the same copy of the loop: a step that has a record,
ran none of the edited functions and whose dependencies all came out as recorded takes its
recorded output; every other step runs on the patched code and is compared with its record. A
query is replayed only when its record is complete: no step shared by two others (the second
could see what the first changed in it), no subquery (the executor keeps subquery plans and
results in caches shared between steps), and the record replays to the pinned result. An edit to
any function that ran around the steps turns the replay off.

    tpcds_memo.py <sources> <target dir>
"""

import hashlib
import marshal
import os
import pickle
import sys
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import tpcds_check  # noqa: E402

TOOL = 3  # a sys.monitoring tool id no debugger, coverage tool or profiler uses
INDEX = "index.marshal"
# what builds the executor's input before any step runs: execute() and the tables it is given.
# An edit there can change what every step reads without running in any step.
AROUND = ("sqlglot/executor/__init__.py", "sqlglot/executor/table.py")


def _blob_digest(blob):
    return hashlib.blake2b(blob, digest_size=16).hexdigest()


def _snapshot(context):
    """The step's output as its dependents receive it: the context's tables, pickled."""
    return pickle.dumps(dict(context.tables), protocol=4)


def _order(plan):
    """The plan's steps, every step after the steps it depends on."""
    dag = plan.dag
    done, out = set(), []

    def visit(node):
        stack = [(node, False)]
        while stack:
            n, ready = stack.pop()
            if n in done:
                continue
            if ready:
                done.add(n)
                out.append(n)
                continue
            stack.append((n, True))
            for d in dag.get(n, ()):
                if d not in done:
                    stack.append((d, False))

    visit(plan.root)
    for n in dag:
        visit(n)
    return out


def _keys(order):
    """Each step's key: its definition (with its dependencies', tpcds_check.plan_digest) and
    the keys of the steps that read it, so two alike steps that feed different ones differ."""
    memo = {}
    plain = {n: tpcds_check.plan_digest(n, memo) for n in order}
    keys = {}
    for n in reversed(order):  # every step after those reading it
        above = sorted(keys[d] for d in n.dependents if d in keys)
        keys[n] = hashlib.blake2b(repr((plain[n], above)).encode("utf-8"), digest_size=16).hexdigest()
    return keys


def _run_step(executor, node, context):
    # PythonExecutor._execute's body for one step, offset included
    from sqlglot import planner

    if isinstance(node, planner.Scan):
        out = executor.scan(node, context)
    elif isinstance(node, planner.Aggregate):
        out = executor.aggregate(node, context)
    elif isinstance(node, planner.Join):
        out = executor.join(node, context)
    elif isinstance(node, planner.Sort):
        out = executor.sort(node, context)
    elif isinstance(node, planner.SetOperation):
        out = executor.set_operation(node, context)
    else:
        raise NotImplementedError
    if node.offset:
        table = out.tables[node.name]
        table.rows = table.rows[node.offset :]
    return out


def _inputs(executor, node, contexts):
    return executor.context(
        {name: table for dep in node.dependencies for name, table in contexts(dep).tables.items()}
    )


class _Lines:
    """The sqlglot code objects that run, segment by segment (each line fires once a segment)."""

    def __init__(self, root):
        self.base = os.path.join(os.path.abspath(root), "sqlglot") + os.sep
        self.ids = {}
        self.table = []  # id -> (path, qualname, first line, last line)
        self.current = set()
        mon = sys.monitoring
        mon.use_tool_id(TOOL, "tpcds_memo")
        mon.register_callback(TOOL, mon.events.LINE, self._line)
        mon.set_events(TOOL, mon.events.LINE)

    def _line(self, code, _line):
        cid = self.ids.get(code)
        if cid is None:
            name = code.co_filename
            if name.startswith(self.base):
                lines = [ln for _s, _e, ln in code.co_lines() if ln is not None]
                cid = len(self.table)
                rel = "sqlglot/" + name[len(self.base) :].replace(os.sep, "/")
                self.table.append((rel, code.co_qualname, code.co_firstlineno, max(lines + [code.co_firstlineno])))
            else:
                cid = -1
            self.ids[code] = cid
        if cid >= 0:
            self.current.add(cid)
        return sys.monitoring.DISABLE

    def take(self):
        out, self.current = self.current, set()
        sys.monitoring.restart_events()
        return out

    def close(self):
        mon = sys.monitoring
        mon.set_events(TOOL, 0)
        mon.register_callback(TOOL, mon.events.LINE, None)
        mon.free_tool_id(TOOL)


def record(root, target):
    if root not in sys.path:
        sys.path.insert(0, root)
    from sqlglot import find_tables, parse_one
    from sqlglot.executor import execute
    from sqlglot.executor.python import PythonExecutor
    from tests.helpers import FIXTURES_DIR, TPCDS_SCHEMA, load_sql_fixture_pairs

    sqls = list(load_sql_fixture_pairs(tpcds_check.FIXTURE))
    lines = _Lines(root)
    frame = set()
    lines.take()
    schema, tables = pickle.loads(pickle.dumps((TPCDS_SCHEMA, tpcds_check._tables(FIXTURES_DIR, TPCDS_SCHEMA))))
    frame |= lines.take()  # the tables' loading
    original_init, original_execute = PythonExecutor.__init__, PythonExecutor._execute
    state = {"rec": None, "busy": False}

    def init(self, *args, **kwargs):
        lines.take()
        original_init(self, *args, **kwargs)
        frame.update(lines.take())

    def loop(self, plan):
        rec = state["rec"]
        if rec is None or state["busy"] or self._outer_scope is not None:
            if rec is not None:
                rec["ok"] = False  # a subquery ran
            return original_execute(self, plan)
        state["busy"] = True
        t_loop = time.process_time()
        try:
            contexts, exclusive = {}, {}
            order = _order(plan)
            keys = _keys(order)
            for node in order:
                context = _inputs(self, node, contexts.__getitem__)
                rec["frame"].update(lines.take())
                t0 = time.process_time()
                out = _run_step(self, node, context)
                cpu = time.process_time() - t0
                codes = lines.take()
                blob = _snapshot(out)
                key = keys[node]
                # only this step's one dependent holds its tables: none is shared with another
                # step, directly or through a table it passes on from its own inputs
                ins = [t for t in context.tables.values()]
                aliased = any(o is i for o in out.tables.values() for i in ins)
                exclusive[node] = len(node.dependents) <= 1 and (
                    not aliased or all(exclusive[d] for d in node.dependencies))
                old = rec["steps"].get(key)
                digests = (old[2] if old else ()) + (_blob_digest(blob),)
                rec["steps"][key] = (tuple(sorted(set(codes) | set(old[0] if old else ()))), cpu + (old[1] if old else 0.0),
                                     digests, exclusive[node] and not old, len(digests),
                                     aliased or bool(old and old[5]))
                rec["blobs"][key] = zlib.compress(blob, 1)
                contexts[node] = out
            rec["root"] = keys[plan.root]
            rec["back"] = time.process_time() - t_loop
            return contexts[plan.root].tables[plan.root.name]
        finally:
            rec["frame"].update(lines.take())
            state["busy"] = False

    def subquery(self, *args, **kwargs):
        if state["rec"] is not None:
            state["rec"]["ok"] = False
        return original_subquery(self, *args, **kwargs)

    original_subquery = PythonExecutor._subquery_table
    PythonExecutor.__init__, PythonExecutor._execute, PythonExecutor._subquery_table = init, loop, subquery
    queries = {}
    os.makedirs(target, exist_ok=True)
    try:
        for k, (meta, sql, _) in enumerate(sqls):
            if not meta.get("execute"):
                continue
            rec = state["rec"] = {"ok": True, "steps": {}, "blobs": {}, "frame": set(), "root": None, "back": 0.0}
            t0 = time.process_time()
            try:
                expression = pickle.loads(pickle.dumps(parse_one(sql)))
                found = {t.name: tables[t.name] for t in find_tables(expression)}
                table = pickle.loads(pickle.dumps(execute(expression, schema=schema, tables=found)))
                rows = tpcds_check._digest((list(table.columns), [tuple(row) for row in table.rows]))
            except Exception as error:  # noqa: BLE001
                print("tpcds_memo: query", k, "failed:", repr(error)[:200], file=sys.stderr)
                rec["ok"] = False
                rows = None
            total = time.process_time() - t0
            state["rec"] = None
            with open(os.path.join(target, "q%d.marshal" % k), "wb") as f:
                marshal.dump(rec["blobs"], f)
            queries[k] = {
                "ok": rec["ok"] and rec["root"] is not None,
                "rows": rows,
                "root": rec["root"],
                "front": max(total - rec["back"], 0.0),
                "back": rec["back"],
                "frame": tuple(sorted(rec["frame"])),
                "steps": rec["steps"],
            }
    finally:
        PythonExecutor.__init__, PythonExecutor._execute = original_init, original_execute
        PythonExecutor._subquery_table = original_subquery
        lines.close()
    index = {"codes": lines.table, "frame": tuple(sorted(frame)), "queries": queries}
    with open(os.path.join(target, INDEX), "wb") as f:
        marshal.dump(index, f)
    return index


class _Fallback(Exception):
    pass


def dirty_codes(codes, ranges):
    """Ids of the recorded code objects whose lines meet any of ``ranges`` ({path: [(lo, hi)]})."""
    out = set()
    for i, (path, _name, first, last) in enumerate(codes):
        for lo, hi in ranges.get(path, ()):
            if lo <= last and first <= hi:
                out.add(i)
                break
    return out


class Replay:
    """Installed in a patched child: PythonExecutor._execute replays the active query's steps."""

    def __init__(self, directory, ranges, index=None):
        self.dir = directory
        if index is None:
            with open(os.path.join(directory, INDEX), "rb") as f:
                index = marshal.load(f)
        self.queries = index["queries"]
        self.dirty = dirty_codes(index["codes"], ranges)
        self.off = bool(self.dirty & set(index["frame"])) or any(p in AROUND for p in ranges)
        self.active = None
        self.busy = False
        self.ran = 0
        self.kept = 0
        self.fallbacks = 0

    def usable(self, k):
        q = self.queries.get(k)
        return (q is not None and q["ok"] and not self.off and not (self.dirty & set(q["frame"])))

    def install(self):
        from sqlglot.executor.python import PythonExecutor

        original = PythonExecutor._execute
        replay = self

        def _execute(executor, plan):
            if replay.active is None or replay.busy or executor._outer_scope is not None:
                return original(executor, plan)
            replay.busy = True
            try:
                return replay.loop(executor, plan)
            except _Fallback:
                replay.fallbacks += 1
            finally:
                replay.busy = False
            return original(executor, plan)

        PythonExecutor._execute = _execute

    def loop(self, executor, plan):
        k = self.active
        steps = self.queries[k]["steps"]
        blobs = {}
        status, contexts = {}, {}
        order = _order(plan)
        keys = _keys(order)

        def context_of(node):
            got = contexts.get(node)
            if got is None:
                if not blobs:
                    with open(os.path.join(self.dir, "q%d.marshal" % k), "rb") as f:
                        blobs.update(marshal.load(f))
                got = contexts[node] = executor.context(pickle.loads(zlib.decompress(blobs[keys[node]])))
            return got

        for node in order:
            meta = steps.get(keys[node])
            if (meta is not None and len(set(meta[2])) == 1 and not self.dirty.intersection(meta[0])
                    and all(status[d] != "changed" for d in node.dependencies)):
                status[node] = "kept"
                self.kept += 1
                continue
            for d in node.dependencies:
                dmeta = steps.get(keys[d])
                if dmeta is not None and dmeta[3]:
                    continue  # its tables are this step's alone
                if dmeta is not None and status[d] == "changed" and not dmeta[5]:
                    continue  # it ran and came out different: every step that reads it runs, on its tables
                # a step that runs would read tables another step holds too, which that one,
                # kept from its record, does not change here: run the query as it is
                raise _Fallback
            out = _run_step(executor, node, _inputs(executor, node, context_of))
            self.ran += 1
            contexts[node] = out
            same = meta is not None and len(set(meta[2])) == 1 and _blob_digest(_snapshot(out)) == meta[2][0]
            status[node] = "same" if same else "changed"
        return context_of(plan.root).tables[plan.root.name]


def estimate(index, ranges, k):
    """Share of query k's build-time cost that a replay under ``ranges`` spends, or None when the
    query cannot be replayed."""
    q = index["queries"].get(k)
    if q is None or not q["ok"] or any(p in AROUND for p in ranges):
        return None
    dirty = dirty_codes(index["codes"], ranges)
    if dirty & set(index["frame"]) or dirty & set(q["frame"]):
        return None
    spent = q["front"] + sum(m[1] for m in q["steps"].values() if dirty.intersection(m[0]))
    spent += 0.002 * len(q["steps"])
    return min(1.0, spent / max(q["front"] + q["back"], 1e-6))


def self_check(root, target):
    """Replays every recorded query with nothing edited; a query whose replay does not give the
    pinned result is marked not replayable. False when the records are unusable as a whole."""
    with open(os.path.join(target, INDEX), "rb") as f:
        index = marshal.load(f)
    replay = Replay(target, {}, index)
    replay.install()
    ks = sorted(k for k, q in index["queries"].items() if q["ok"])
    got = tpcds_check.fingerprints(root, ks, replay=replay)
    bad = [k for k in ks if got[k][0] != index["queries"][k]["rows"]]
    for k in bad:
        index["queries"][k]["ok"] = False
    with open(os.path.join(target, INDEX), "wb") as f:
        marshal.dump(index, f)
    print("tpcds_memo: %d queries replayable, %d replayed wrong; %d steps kept, %d run, %d fallbacks" % (
        len(ks) - len(bad), len(bad), replay.kept, replay.ran, replay.fallbacks), file=sys.stderr)
    return len(bad) < len(ks)


def main():
    root, target = sys.argv[1], sys.argv[2]
    try:
        record(root, target)
        if not self_check(root, target):
            raise RuntimeError("no query replays to its pinned result")
    except Exception as error:  # noqa: BLE001
        print("tpcds_memo: no step records:", repr(error)[:300], file=sys.stderr)
        try:
            os.unlink(os.path.join(target, INDEX))
        except OSError:
            pass


if __name__ == "__main__":
    main()
