"""Generates synthetic single-edit patches against the pinned sqlglot sources.

    python eval/mutants.py <sqlglot checkout> <out dir> <count> [--seed N] [--where any|import|runtime]

The operators mirror the edits seen in the sample commits: comparison swaps (< <=, == !=,
is ==, is None == None), and/or swaps, dropping ``not``, integers +-1, True/False flips,
comment lines added to Python files and junk tokens added to SQL fixture lines. Each patch is a
unified diff with three lines of context, written as <out dir>/<id>.patch, plus manifest.json
describing what was changed.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "profiling"))

from source_index import IMPORT, MIXED, RUNTIME, index_source  # noqa: E402

COMMENTS = [
    "# Cheap check first",
    "# Keep in sync with the generator",
    "# Intentionally not cached",
    "# Kept for backwards compatibility",
    "# Mirrors the behavior of the reference implementation",
]
JUNK = ["NULL", "+ 42", "x", "AND", ",", "1", "FROM", "(", ")"]
CMP_SWAP = {"<": "<=", "<=": "<", ">": ">=", ">=": ">", "==": "!=", "!=": "==", " is not ": " != ", " is ": " == "}


def sites_in(rel, source):
    """(line, col0, col1, replacement, label) edits of one token or keyword."""
    lines = source.split("\n")
    tree = ast.parse(source)
    out = []

    def seg(node_a, node_b):
        """Text between two nodes when both sit on the same line."""
        if node_a.end_lineno != node_b.lineno:
            return None
        return node_a.end_lineno, node_a.end_col_offset, node_b.col_offset

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            where = seg(node.left, node.comparators[0])
            if not where:
                continue
            ln, c0, c1 = where
            text = lines[ln - 1][c0:c1]
            for old, new in CMP_SWAP.items():
                if old.strip() in ("is", "is not"):
                    if text == old:
                        out.append((ln, c0, c1, new, "cmp"))
                        break
                elif text.strip() == old:
                    i = lines[ln - 1].index(old, c0)
                    out.append((ln, i, i + len(old), new, "cmp"))
                    break
        elif isinstance(node, ast.BoolOp) and len(node.values) >= 2:
            where = seg(node.values[0], node.values[1])
            if not where:
                continue
            ln, c0, c1 = where
            text = lines[ln - 1][c0:c1]
            word = "and" if isinstance(node.op, ast.And) else "or"
            if text.strip() == word:
                i = lines[ln - 1].index(word, c0)
                out.append((ln, i, i + len(word), "or" if word == "and" else "and", "boolop"))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            ln, c0 = node.lineno, node.col_offset
            if lines[ln - 1][c0 : c0 + 4] == "not ":
                out.append((ln, c0, c0 + 4, "", "not"))
        elif isinstance(node, ast.Constant) and node.lineno == node.end_lineno:
            ln, c0, c1 = node.lineno, node.col_offset, node.end_col_offset
            text = lines[ln - 1][c0:c1]
            if node.value is True and text == "True":
                out.append((ln, c0, c1, "False", "bool"))
            elif node.value is False and text == "False":
                out.append((ln, c0, c1, "True", "bool"))
            elif type(node.value) is int and text.isdigit():
                out.append((ln, c0, c1, str(node.value + 1), "int"))
    return out


def unified(rel, old_lines, new_lines, start, removed, added):
    """A git-style diff replacing ``removed`` old lines at ``start`` (1-based) with ``added``."""
    ctx0 = max(1, start - 3)
    ctx1 = min(len(old_lines), start + removed - 1 + 3)
    before = old_lines[ctx0 - 1 : start - 1]
    after = old_lines[start - 1 + removed : ctx1]
    body = [" " + l for l in before]
    body += ["-" + l for l in old_lines[start - 1 : start - 1 + removed]]
    body += ["+" + l for l in added]
    body += [" " + l for l in after]
    old_count = len(before) + removed + len(after)
    new_count = len(before) + len(added) + len(after)
    old_start = ctx0 if old_count else start - 1
    head = "diff --git a/%s b/%s\n--- a/%s\n+++ b/%s\n@@ -%d,%d +%d,%d @@\n" % (
        rel, rel, rel, rel, old_start, old_count, ctx0, new_count)
    return head + "\n".join(body) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("count", type=int)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--where", default="any", choices=["any", "import", "runtime"])
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    code_sites = []
    py_files = []
    for base, dirs, names in os.walk(os.path.join(args.src, "sqlglot")):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in names:
            if name.endswith(".py"):
                rel = os.path.relpath(os.path.join(base, name), args.src)
                source = open(os.path.join(args.src, rel), encoding="utf-8").read()
                ix = index_source(source)
                py_files.append(rel)
                for site in sites_in(rel, source):
                    kind = ix["kind"][site[0] - 1]
                    where = "runtime" if kind == RUNTIME else "import" if kind in (IMPORT, MIXED) else "other"
                    if args.where != "any" and where != args.where:
                        continue
                    code_sites.append((rel, where) + site)
    fixtures = []
    for base, _dirs, names in os.walk(os.path.join(args.src, "tests", "fixtures")):
        for name in names:
            if name.endswith(".sql"):
                fixtures.append(os.path.relpath(os.path.join(base, name), args.src))

    manifest = {}
    for i in range(args.count):
        cid = "m%04d" % (args.seed * 10000 + i)
        roll = rng.random()
        if args.where != "any" or roll < 0.7:
            rel, where, ln, c0, c1, new, label = rng.choice(code_sites)
            lines = open(os.path.join(args.src, rel), encoding="utf-8").read().split("\n")
            old = lines[ln - 1]
            patch = unified(rel, lines, None, ln, 1, [old[:c0] + new + old[c1:]])
            info = {"kind": "code", "file": rel, "line": ln, "op": label, "where": where}
        elif roll < 0.85:
            rel = rng.choice(fixtures)
            lines = open(os.path.join(args.src, rel), encoding="utf-8").read().split("\n")
            candidates = [k for k, l in enumerate(lines, 1) if l.strip() and not l.startswith("#") and not l.startswith("--")]
            if not candidates:
                continue
            ln = rng.choice(candidates)
            words = lines[ln - 1].split(" ")
            pos = rng.randrange(len(words) + 1)
            words.insert(pos, rng.choice(JUNK))
            patch = unified(rel, lines, None, ln, 1, [" ".join(words)])
            info = {"kind": "fixture", "file": rel, "line": ln}
        else:
            rel = rng.choice(py_files)
            lines = open(os.path.join(args.src, rel), encoding="utf-8").read().split("\n")
            ln = rng.randrange(1, len(lines))
            indent = lines[ln - 1][: len(lines[ln - 1]) - len(lines[ln - 1].lstrip())]
            patch = unified(rel, lines, None, ln, 0, [indent + rng.choice(COMMENTS)])
            info = {"kind": "comment", "file": rel, "line": ln}
        with open(os.path.join(args.out, cid + ".patch"), "w", encoding="utf-8") as f:
            f.write(patch)
        manifest[cid] = info
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("wrote %d patches to %s" % (len(manifest), args.out))


if __name__ == "__main__":
    main()
