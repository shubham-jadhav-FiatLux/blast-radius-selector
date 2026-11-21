"""A scaled re-check of tests/test_parser.py::TestParser::test_max_nodes (28.8 s).

The test parses ``SELECT x=x=...=x, ...`` (10,000 identical columns, 100 comparisons each, two
million characters) twice and expects a ParseError about the node limit each time, plus four
small parses. Almost all of its time is tokenizing those two million characters and building a
million nodes before the limit trips.

The re-check runs the test's small assertions as written and its two big parses on the same
shape with fewer columns and proportionally smaller limits, and asks for exactly the error
messages (limit, line, column and highlighted context) that the pinned commit gives, from a
parse that scales like the pinned one (CPU per column within a small factor, measured at two
sizes). Agreement at small sizes is evidence, not proof, for the full size: the input repeats
one column, so only code that behaves differently after hundreds of identical columns, or that
grows faster than linearly, could tell the two apart; the timing condition catches the latter.

    max_nodes_check.py --reference <sources> <out.json>   (build time)
"""

import json
import sys
import time
import unittest  # noqa: F401  sqlglot's strict argument checks are on under tests

TEST = "tests/test_parser.py::TestParser::test_max_nodes"
DEPTH = 100
# columns, the "1M" limit and the "50k" limit scaled with them: the test's 1,000,000 is its
# 10,000 columns times the 100 comparisons each (only those count), so it trips at the very end
# of the input, and 50,000 trips a twentieth of the way in
SIZES = ((40, 40 * DEPTH, 40 * 5), (200, 200 * DEPTH, 200 * 5))


def _wide(width):
    expr = "x" + "=x" * DEPTH
    return "SELECT " + ",".join([expr] * width)


def observe(root):
    """What the scaled test sees: small parses (types or errors), the big parses' messages and
    the CPU each size took."""
    if root not in sys.path:
        sys.path.insert(0, root)
    from sqlglot import exp, parse_one
    from sqlglot.errors import ParseError

    out = {"small": [], "big": [], "seconds": []}

    def kind(sql, **kwargs):
        try:
            return "ok:" + type(parse_one(sql, **kwargs)).__name__
        except ParseError as e:
            return "ParseError:" + str(e)

    cols = ",".join("col%d" % i for i in range(500))
    out["small"] = [
        kind("SELECT 1", max_nodes=1_000_000),
        kind("SELECT %s FROM t" % cols, max_nodes=1_000_000),
        kind("SELECT " + ",".join("x=%d" % i for i in range(100)), max_nodes=5),
        kind("SELECT 1, 2, 3"),
    ]
    assert exp is not None
    for width, high, low in SIZES:
        sql = _wide(width)
        started = time.process_time()
        for limit in (high, low):
            out["big"].append(kind(sql, max_nodes=limit))
        out["seconds"].append(time.process_time() - started)
    return out


def same_as(reference, got, slack=2.0):
    """True if the patched sources behaved like the pinned ones on the scaled test: the same
    results and messages, and the larger size costing about as much more than the smaller one
    as it did at the pinned commit (five times the columns: a parse that became quadratic would
    cost twenty-five times more). Absolute times are not compared: the machine may differ."""
    if got["small"] != reference["small"] or got["big"] != reference["big"]:
        return False
    if not reference["small"][0].startswith("ok:Select") or not all(b.startswith("ParseError:") for b in got["big"]):
        return False
    ref_small, ref_large = reference["seconds"]
    small, large = got["seconds"]
    return large / max(small, 1e-3) <= slack * ref_large / max(ref_small, 1e-3)


def main():
    if sys.argv[1] != "--reference":
        raise SystemExit(__doc__)
    root, target = sys.argv[2], sys.argv[3]
    try:
        first = observe(root)
        again = observe(root)
        if first["small"] != again["small"] or first["big"] != again["big"]:
            raise RuntimeError("not reproducible")
        first["seconds"] = [min(a, b) for a, b in zip(first["seconds"], again["seconds"])]
        with open(target, "w") as f:
            json.dump(first, f)
    except Exception as e:  # noqa: BLE001
        print("max_nodes_check: no reference:", repr(e), file=sys.stderr)


if __name__ == "__main__":
    main()
