"""Default-use profiling: which tests call a function without passing an argument it has a
default for.

An edit to a default value (``end: int = 0`` -> ``end: int = 1``) can only change a call that
leaves that argument out. Most calls in sqlglot pass everything explicitly, so "tests that ran
the function" is often far more than "tests that ever used the default". While profiling, every
literal default of a sqlglot function is replaced by a sentinel, and the function starts with

    if end is __dflt_sentinel__:
        __dflt_used__(site)
        end = 0

so the function behaves exactly as before (a literal evaluates to the same value at call time as
at definition time), and the current test is noted as one that used that default. Only literal
defaults (numbers, strings, booleans, None, and their negations) are rewritten; nothing in
sqlglot reads default values back through introspection, only parameter names.

``SITES[site]`` is (relpath, line, col0, col1, parameter, def line) of the default expression.
"""

from __future__ import annotations

import ast

SITES = []
SENTINEL = object()


def _literal(node):
    if isinstance(node, ast.Constant):
        return True
    return isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)) and isinstance(node.operand, ast.Constant)


class _Rewriter(ast.NodeTransformer):
    def __init__(self, rel):
        self.rel = rel

    def _rewrite(self, node):
        self.generic_visit(node)
        args = node.args
        pairs = []
        positional = args.posonlyargs + args.args
        for param, default in zip(positional[len(positional) - len(args.defaults):], args.defaults):
            pairs.append((param, default, "defaults"))
        for param, default in zip(args.kwonlyargs, args.kw_defaults):
            if default is not None:
                pairs.append((param, default, "kw_defaults"))
        checks = []
        for param, default, where in pairs:
            if not _literal(default) or default.lineno != default.end_lineno:
                continue
            site = len(SITES)
            SITES.append((self.rel, default.lineno, default.col_offset, default.end_col_offset, param.arg, node.lineno))
            sentinel = ast.copy_location(ast.Name(id="__dflt_sentinel__", ctx=ast.Load()), default)
            values = getattr(args, where)
            values[values.index(default)] = sentinel
            test = ast.Compare(left=ast.Name(id=param.arg, ctx=ast.Load()), ops=[ast.Is()],
                               comparators=[ast.Name(id="__dflt_sentinel__", ctx=ast.Load())])
            body = [
                ast.Expr(ast.Call(func=ast.Name(id="__dflt_used__", ctx=ast.Load()), args=[ast.Constant(site)], keywords=[])),
                ast.Assign(targets=[ast.Name(id=param.arg, ctx=ast.Store())], value=default),
            ]
            checks.append(ast.If(test=test, body=body, orelse=[]))
        if checks:
            start = 0
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                start = 1  # keep the docstring first
            for check in checks:
                ast.copy_location(check, node)
                ast.fix_missing_locations(check)
            node.body[start:start] = checks
        return node

    visit_FunctionDef = _rewrite
    visit_AsyncFunctionDef = _rewrite


def rewrite(source, rel, filename):
    tree = ast.parse(source, filename)
    tree = _Rewriter(rel).visit(tree)
    ast.fix_missing_locations(tree)
    return compile(tree, filename, "exec", dont_inherit=True)


def install(used):
    """Puts the helpers into builtins; ``used(site)`` notes a default used in the current context."""
    import builtins

    builtins.__dflt_sentinel__ = SENTINEL
    builtins.__dflt_used__ = used
