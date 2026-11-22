"""Folds a PROFILE_CASES=1 profile into the per-case map, streaming (the profile is large).

For each test that walks a SQL fixture file case by case: the fixture files and case counts,
the lines it ran outside any case, and for each line that at most CAP of its cases ran, those
cases. A line more cases ran cannot narrow a re-check much and is left out (as unknown).

    build_cases.py <profile.txt.gz> <out.marshal>
"""

import gzip
import marshal
import sys
from collections import defaultdict

CAP = 64


def main():
    path, target = sys.argv[1], sys.argv[2]
    files = {}
    with gzip.open(path, "rt") as f:
        for raw in f:
            if raw.startswith("F "):
                fid, rel = raw[2:].rstrip("\n").split("\t", 1)
                files[int(fid)] = rel
    per = {}
    with gzip.open(path, "rt") as f:
        for raw in f:
            if not raw.startswith("C "):
                continue
            ctx, _, payload = raw[2:].rstrip("\n").partition("\t")
            nodeid, _, phase = ctx.rpartition("|")
            rec = per.get(nodeid)
            if rec is None:
                rec = per[nodeid] = {"files": {}, "count": defaultdict(int), "outside": set(), "cases": {}}
            pairs = []
            for item in payload.split(","):
                fid, line = item.split(":")
                line = int(line)
                if line > 0:
                    pairs.append((files[int(fid)], line))
            if "#c" not in phase:
                rec["outside"].update(pairs)
                continue
            fname, _, k = phase.split("#c", 1)[1].rpartition(":")
            fid = rec["files"].setdefault(fname, len(rec["files"]))
            rec["count"][fid] = max(rec["count"][fid], int(k) + 1)
            case = (fid, int(k))
            cases = rec["cases"]
            for key in pairs:
                got = cases.get(key, ())
                if got is None:
                    continue
                if len(got) >= CAP:
                    cases[key] = None  # too many to narrow
                    continue
                cases[key] = got + (case,) if got else (case,)
    out = {}
    for nodeid, rec in per.items():
        if not rec["files"]:
            continue
        lines = defaultdict(dict)
        for (rel, ln), cs in rec["cases"].items():
            if cs is None or (rel, ln) in rec["outside"]:
                continue
            lines[rel][ln] = tuple(sorted(set(cs)))
        outside = defaultdict(list)
        for rel, ln in rec["outside"]:
            outside[rel].append(ln)
        out[nodeid] = {"files": dict(rec["files"]), "total": sum(rec["count"].values()),
                       "lines": dict(lines), "outside": {r: sorted(v) for r, v in outside.items()}}
    with open(target, "wb") as f:
        marshal.dump(out, f)
    print("build_cases: %d fixture tests" % len(out), file=sys.stderr)


if __name__ == "__main__":
    main()
