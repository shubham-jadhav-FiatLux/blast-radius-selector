"""Compares a selector output with labels for whichever commits are labelled so far.

    python eval/check_labels.py <set dir with tests.json> <labels.json> <output.jsonl>
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from blast_select import split  # noqa: E402


def main():
    root, labels_path, output = sys.argv[1:4]
    tests = json.load(open(os.path.join(root, "tests.json")))["tests"]
    ms = {t["number"]: int(round(t["seconds"] * 1000)) for t in tests}
    name = {t["number"]: t["id"] for t in tests}
    labels = json.load(open(labels_path))["commits"]
    manifest = {}
    mpath = os.path.join(root, "commits", "manifest.json")
    if os.path.exists(mpath):
        manifest = json.load(open(mpath))
    rows = {json.loads(l)["commit"]: json.loads(l)["workers"] for l in open(output) if l.strip()}

    def cost(workers):
        workers = [w for w in workers if w]
        return max(sum(ms[t] for t in w) for w in workers) + 2000 * len(workers) if workers else 0

    total = oracle = misses = 0
    for cid in sorted(labels):
        failing = labels[cid]["failing"]
        workers = rows.get(cid, [])
        chosen = {t for w in workers for t in w}
        missed = [t for t in failing if t not in chosen]
        c = cost(workers) + (300_000 if missed else 0)
        o = cost(split(failing, ms))
        total += c
        oracle += o
        misses += bool(missed)
        info = manifest.get(cid, {})
        print("%s %-8s %-7s %-6s fail=%-4d sel=%-5d cost=%7.2f oracle=%6.2f %s %s" % (
            cid, info.get("kind", "?"), info.get("where", ""), info.get("op", ""), len(failing), len(chosen),
            c / 1000, o / 1000, info.get("file", ""), ("MISS " + ",".join(name[t] for t in missed[:2])) if missed else ""))
    print("TOTAL %.1f s  oracle %.1f s  misses %d  (%d labelled)" % (total / 1000, oracle / 1000, misses, len(labels)))


if __name__ == "__main__":
    main()
