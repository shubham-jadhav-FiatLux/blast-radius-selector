"""Labels patches by running the whole sqlglot suite on each one, the way the track does.

    python eval/label.py <sqlglot checkout> <patch dir> <tests.json> <labels.json> [--image IMG] [--jobs N]

For every patch: copy of the pinned tree with the patch applied, full suite in the labelling
image (process pools and all), failing tests mapped to their numbers in tests.json. Results are
appended to <labels.json> in the kit's answers format, so an interrupted run can be resumed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from blast_select import parse_patch  # noqa: E402

PLUGINS = os.path.abspath(os.path.join(HERE, "..", "profiling"))


def apply(tree, patch_text):
    """Applies a patch to the copy in place; returns {path: original text} to restore."""
    saved = {}
    for change in parse_patch(patch_text):
        path = os.path.join(tree, change.path)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        saved[path] = text
        lines = text.split("\n")
        for block in sorted(change.blocks, key=lambda b: b.start, reverse=True):
            i = block.start - 1
            if lines[i : i + len(block.old)] != block.old:
                raise ValueError("patch does not apply to %s at line %d" % (change.path, block.start))
            lines[i : i + len(block.old)] = block.new
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    return saved


def run_suite(tree, out_dir, image, jobs, targets=("tests",)):
    out = os.path.join(out_dir, "outcomes.json")
    if os.path.exists(out):
        os.remove(out)
    parallel = ["-n", str(jobs)] if jobs > 1 else ["-p", "no:xdist"]
    # Pinning the container to as many CPUs as it may use keeps process pools (sized by
    # os.process_cpu_count()) from oversubscribing the machine.
    cpus = max(jobs, 2)
    cmd = [
        "docker", "run", "--rm", "--cpuset-cpus", "0-%d" % (cpus - 1), "--memory", "5g", "--memory-swap", "5g",
        "-v", "%s:/src:ro" % tree, "-v", "%s:/plugins:ro" % PLUGINS, "-v", "%s:/out" % out_dir,
        "-e", "PYTHONPATH=/src:/plugins", "-e", "OUTCOME_OUT=/out/outcomes.json", "-w", "/src", image,
        "python", "-m", "pytest", "-q", "-p", "outcome_plugin", "-p", "no:cacheprovider",
        *parallel, "--timeout", "600", *targets,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3600)
    with open(out) as f:
        return json.load(f)


def failing_ids(result, by_file_ids):
    failed = {n for n, r in result["results"].items() if r["failed"]}
    for nodeid in result["collect_errors"]:
        failed.update(by_file_ids.get(nodeid.split("::")[0], []))
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("patches")
    ap.add_argument("tests")
    ap.add_argument("labels")
    ap.add_argument("--image", default="blast-radius-profile")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    tests = json.load(open(args.tests))["tests"]
    number = {t["id"]: t["number"] for t in tests}
    by_file_ids = {}
    for t in tests:
        by_file_ids.setdefault(t["file"], []).append(t["id"])
    labels = {"commits": {}}
    if os.path.exists(args.labels):
        labels = json.load(open(args.labels))
    manifest = {}
    if os.path.exists(os.path.join(args.patches, "manifest.json")):
        manifest = json.load(open(os.path.join(args.patches, "manifest.json")))

    work = tempfile.mkdtemp(prefix="label-")
    tree = os.path.join(work, "src")
    out_dir = os.path.join(work, "out")
    os.makedirs(out_dir)
    os.chmod(out_dir, 0o777)
    shutil.copytree(args.src, tree, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    try:
        for name in sorted(os.listdir(args.patches)):
            if not name.endswith(".patch"):
                continue
            cid = name[:-6]
            if cid in labels["commits"]:
                continue
            text = open(os.path.join(args.patches, name), encoding="utf-8").read()
            saved = apply(tree, text)
            try:
                result = run_suite(tree, out_dir, args.image, args.jobs)
                failed = failing_ids(result, by_file_ids)
                # A parallel run on a busy machine can lose a worker (process pools, memory);
                # a failure only counts if it happens again with the failing tests run alone.
                unconfirmed = set()
                if failed and len(failed) <= 200:
                    again = run_suite(tree, out_dir, args.image, 1, sorted(failed))
                    confirmed = failing_ids(again, by_file_ids)
                    unconfirmed = failed - confirmed
                    failed = confirmed
            finally:
                for path, original in saved.items():
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(original)
            failing = {number[n] for n in failed if n in number}
            kind = manifest.get(cid, {}).get("kind", "code")
            labels["commits"][cid] = {"failing": sorted(failing), "kind": kind}
            if unconfirmed:
                labels["commits"][cid]["unconfirmed"] = sorted(number[n] for n in unconfirmed if n in number)
            with open(args.labels, "w") as f:
                json.dump(labels, f)
            print("%s %-8s %4d failing (%d unconfirmed)  %s" % (
                cid, kind, len(failing), len(unconfirmed), manifest.get(cid, {}).get("file", "")), flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
