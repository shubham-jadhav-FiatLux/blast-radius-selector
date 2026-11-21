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

FIXTURE = "optimizer/tpc-ds/tpc-ds.sql"


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
        schema, tables = pickle.loads(pickle.dumps((TPCDS_SCHEMA, _tables(FIXTURES_DIR, TPCDS_SCHEMA))))
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


def main():
    """Writes the reference fingerprints and timings: tpcds_check.py <sources> <out.json>"""
    import json
    import time

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
