"""Per-commit cost of a selector output against labelled commits.

    python eval/breakdown.py <sample dir> <answers.json> <output.jsonl>

Costs follow SCORING.md: busiest worker + 2 s per worker, +300 s when a failing test is left out.
The oracle column is the cost of running exactly the failing tests, split as well as possible.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from blast_select import split  # noqa: E402


def cost(workers, ms):
    workers = [w for w in workers if w]
    if not workers:
        return 0
    return max(sum(ms[t] for t in w) for w in workers) + 2000 * len(workers)


def main():
    sample, answers_path, output = sys.argv[1:4]
    tests = json.load(open(os.path.join(sample, "tests.json")))["tests"]
    ms = {t["number"]: int(round(t["seconds"] * 1000)) for t in tests}
    names = {t["number"]: t["id"] for t in tests}
    answers = json.load(open(answers_path))["commits"]
    rows = {}
    for line in open(output):
        line = line.strip()
        if line:
            d = json.loads(line)
            rows[d["commit"]] = d["workers"]
    total = oracle = 0
    misses = 0
    for cid in sorted(answers):
        a = answers[cid]
        workers = rows.get(cid, [])
        chosen = {t for w in workers for t in w}
        missed = [t for t in a["failing"] if t not in chosen]
        c = cost(workers, ms) + (300_000 if missed else 0)
        o = cost(split(a["failing"], ms), ms)
        total += c
        oracle += o
        misses += bool(missed)
        print(
            "%-5s %-8s fail=%-3d sel=%-5d k=%d cost=%7.2f oracle=%6.2f %s"
            % (cid, a["kind"], len(a["failing"]), len(chosen), len(workers), c / 1000, o / 1000,
               ("MISS " + ",".join(names[t].split("::")[-1] for t in missed[:3])) if missed else "")
        )
    print("TOTAL %.1f s  oracle %.1f s  misses %d  (%d commits)" % (total / 1000, oracle / 1000, misses, len(answers)))


if __name__ == "__main__":
    main()
