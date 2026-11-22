"""Profile bounded sets of exact integer operands in isolated light unittest runs."""
import ast
import enum
import builtins
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import marshal
import multiprocessing
import os
from pathlib import Path
import sys
import unittest
import zlib

from blast_select import load_map, RUNTIME
from infection import OPS

from functools import lru_cache

sites = {}
observed = {}
suite = {}
SYMBOL = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!=",
          ast.Is: "is", ast.IsNot: "is not"}


def observe(site, value):
    values = observed.setdefault(site, set())
    if values is not None:
        if type(value) is not int or value.bit_length() > 63:
            observed[site] = None
        else:
            values.add(value)
            if len(values) > 32:
                observed[site] = None
    return value


@lru_cache(maxsize=128)
def relation_mask(op, result, equal, identical, order):
    outcomes = {"==": equal, "!=": not equal, "is": identical, "is not": not identical}
    if order is not None:
        outcomes.update({"<": order < 0, "<=": order <= 0, ">": order > 0, ">=": order >= 0})
    return sum(1 << i for i, name in enumerate(OPS) if outcomes.get(name) is result)


def compare(site, op, left, right):
    try:
        result = OPS[op](left, right)
    except BaseException:
        observed[site] = None
        raise
    if observed.get(site, True) is not None:
        primitives = (str, int, bool, type(None))
        singleton = lambda v: v is None or isinstance(v, enum.Enum)
        # an enum member or None is a singleton: equality and identity agree on it
        objects = (singleton(left) and (singleton(right) or type(right) in primitives)) or (
            singleton(right) and type(left) in primitives)
        if not objects and (type(left) not in primitives or type(right) not in primitives):
            observed[site] = None
        else:
            order = None
            if (type(left) is type(right) is str or
                    type(left) in (int, bool) and type(right) in (int, bool)):
                order = (left > right) - (left < right)
            facts = relation_mask(op, result, left == right, left is right, order)
            # Interning can change when source is recompiled. Only singleton identity or
            # unequal primitive values can support a stable equivalence involving "is".
            if left == right and type(left) not in (bool, type(None)) and not isinstance(left, enum.Enum):
                if op.startswith("is"):
                    observed[site] = None
                    return result
                facts &= ~((1 << tuple(OPS).index("is")) | (1 << tuple(OPS).index("is not")))
            observed.setdefault(site, set()).add(facts)
    return result


class Instrument(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != "sqlglot" and not fullname.startswith("sqlglot."):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec and spec.origin and spec.origin.endswith(".py"):
            spec.loader = Loader(fullname, spec.origin)
        return spec


class Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        file = Path(self.path)
        rel = file.relative_to(root).as_posix()
        source = file.read_text()
        tree = ast.parse(source)
        lines = source.splitlines()
        info = files.get(rel, {})
        class Rewrite(ast.NodeTransformer):
            def visit_Compare(self, n):
                self.generic_visit(n)
                return instrument(n)

        def instrument(n):
            if not isinstance(n, ast.Compare) or len(n.ops) != 1 or n.lineno != n.end_lineno:
                return n
            op = SYMBOL.get(type(n.ops[0]))
            a, b = n.left, n.comparators[0]
            if op not in OPS or not info or info["kind"][n.lineno-1] != RUNTIME:
                return n
            line = lines[n.lineno-1]
            if not line.isascii() or line[a.end_col_offset:b.col_offset].strip() != op:
                return n
            left = isinstance(a, ast.Constant)
            lit, expr = (a,b) if left else (b,a)
            start = line.index(op, a.end_col_offset, b.col_offset)
            if (not isinstance(lit, ast.Constant) or type(lit.value) is not int or
                    isinstance(expr, ast.Constant) or op in ("is", "is not")):
                key = (rel, n.lineno, start, start + len(op), None, None, False)
                return ast.copy_location(ast.Call(ast.Name("__blast_compare__", ast.Load()),
                    [ast.Constant(key), ast.Constant(op), a, b], []), n)
            key = (rel, n.lineno, start, start+len(op), lit.col_offset, lit.end_col_offset, left)
            wrapped = ast.copy_location(ast.Call(ast.Name("__blast_integer__", ast.Load()),
                                                  [ast.Constant(key), expr], []), expr)
            if left:
                n.comparators[0] = wrapped
            else:
                n.left = wrapped
            return n

        Rewrite().visit(tree)
        return compile(ast.fix_missing_locations(tree), self.path, "exec", dont_inherit=True)


def profile(index):
    observed.clear()
    result = unittest.TestResult()
    unittest.TestLoader().loadTestsFromName(suite[index]).run(result)
    if result.testsRun != 1 or not result.wasSuccessful() or result.skipped or result.expectedFailures:
        return {}
    return {k: tuple(sorted(v)) for k,v in observed.items() if v}


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    blob = load_map(Path(sys.argv[2]).read_bytes())
    files = blob["files"]
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.meta_path.insert(0, Instrument())
    builtins.__blast_integer__ = observe
    builtins.__blast_compare__ = compare
    for i, test in enumerate(blob["tests"]):
        parts = test.split("::")
        if len(parts) != 3 or not parts[0].startswith("tests/") or parts[0] == "tests/test_sqlglotc.py":
            continue
        if parts[-1] in ("test_max_nodes", "test_lazy_load", "test_execute_tpcds", "test_generator_memory"):
            continue
        module = parts[0][:-3].replace("/", ".")
        try:
            importlib.import_module(module)
        except ImportError:
            continue
        suite[i] = ".".join([module, *parts[1:]])
    # Fork each test from the same warm parent. No preceding test can populate its caches.
    with multiprocessing.get_context("fork").Pool(4, maxtasksperchild=1) as pool:
        for i, observations in zip(suite, pool.imap(profile, suite, chunksize=1)):
            for key, values in observations.items():
                groups = sites.setdefault(key, {})
                groups[values] = groups.get(values, 0) | (1 << i)
    result = {}
    for (path, line, *span), groups in sites.items():
        result.setdefault((path,line), []).append((*span, groups))
    Path(sys.argv[3]).write_bytes(zlib.compress(marshal.dumps(result)))
    print(f"infection: {len(sites)} observed sites across {len(suite)} attempted tests", file=sys.stderr)
