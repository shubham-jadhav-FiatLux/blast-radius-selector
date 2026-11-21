"""Weak mutation profiling: which tests would see a comparison come out differently.

A test can only fail on a mutant if the mutated code computes something different for it. For
comparison and boolean operator swaps (and dropped ``not``s) that is cheap to observe directly: while profiling, every
``a < b`` inside a sqlglot function is rewritten into a call that evaluates ``a < b`` (the
result the program uses) and also the swapped operators (``a <= b``), and notes whether any of
them would have come out differently in the current context. Same for ``a is b`` against
``a == b``, an integer constant against its neighbours, and ``x and y`` against ``x or y``
where only the truth value matters.

The program's behaviour is unchanged: operands are evaluated once, in order, and the original
result (or exception) is what the rewritten expression produces.

Sites are numbered per process; ``SITES[site]`` is (relpath, line, col0, col1, operator text,
[variant texts]) where col0..col1 is the operator (or constant) in the source line, and the
variant texts are what a mutation would put there.
"""

from __future__ import annotations

import ast
import operator

OPS = {
    ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!=",
    ast.Is: "is", ast.IsNot: "is not", ast.In: "in", ast.NotIn: "not in",
}
FUNCS = {
    "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge,
    "==": operator.eq, "!=": operator.ne, "is": operator.is_, "is not": operator.is_not,
    "in": lambda a, b: a in b, "not in": lambda a, b: a not in b,
}
# The swaps worth observing. Most change the result only for some operand values; the
# complements (== and !=, in and not in, < and >=, ...) change it every time the comparison is
# evaluated, so for them it is enough to note that it ran (a statement can stop before it).
SWAPS = {
    "<": ["<=", ">="], "<=": ["<", ">"], ">": [">=", "<="], ">=": [">", "<"],
    "is": ["==", "is not"], "is not": ["!=", "is"], "==": ["is", "!="], "!=": ["is not", "=="],
    "in": ["not in"], "not in": ["in"],
}
COMPLEMENT = {("<", ">="), ("<=", ">"), (">", "<="), (">=", "<"), ("is", "is not"), ("is not", "is"),
              ("==", "!="), ("!=", "=="), ("in", "not in"), ("not in", "in")}
ALWAYS = object()  # evaluator marker: the swap always changes the result

SITES = []  # site -> (rel, line, col0, col1, text, [variant texts])
_DETAIL = []  # site -> (boolean context, op function, [variant evaluators])


def _bool_context_nodes(tree):
    """Expressions whose truth value is all the program uses."""
    marked = set()

    def mark(node):
        marked.add(id(node))
        if isinstance(node, ast.BoolOp):
            for value in node.values:
                mark(value)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            mark(node.operand)

    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While, ast.IfExp, ast.Assert)):
            mark(node.test)
        elif isinstance(node, ast.comprehension):
            for cond in node.ifs:
                mark(cond)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            mark(node.operand)
        elif type(node).__name__ == "match_case" and node.guard is not None:
            mark(node.guard)
    return marked


def _const_int(node):
    return isinstance(node, ast.Constant) and type(node.value) is int


class _Rewriter(ast.NodeTransformer):
    def __init__(self, rel, lines, bool_nodes):
        self.rel = rel
        self.lines = lines
        self.bool_nodes = bool_nodes
        self.depth = 0  # inside a function body

    # only code inside functions runs per test; module and class bodies run once at import
    def visit_FunctionDef(self, node):
        node.decorator_list = [self.visit(d) for d in node.decorator_list]
        self.depth += 1
        node.body = [self.visit(s) for s in node.body]
        self.depth -= 1
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        self.depth += 1
        node.body = self.visit(node.body)
        self.depth -= 1
        return node

    def _op_columns(self, left, right, text):
        if left.end_lineno != right.lineno:
            return None
        line = self.lines[left.end_lineno - 1]
        between = line[left.end_col_offset : right.col_offset]
        if text in ("is", "is not", "in", "not in"):
            if " ".join(between.split()) != text:
                return None
            start = left.end_col_offset + between.index(text.split()[0])
            if text in ("is not", "not in"):
                end = start + len(between.strip())
                return left.end_lineno, start, end
        else:
            if between.strip() != text:
                return None
            start = left.end_col_offset + between.index(text)
        return left.end_lineno, start, start + len(text)

    def visit_Compare(self, node):
        self.generic_visit(node)
        if self.depth == 0 or len(node.ops) != 1 or type(node.ops[0]) not in OPS:
            return node
        text = OPS[type(node.ops[0])]
        left, right = node.left, node.comparators[0]
        boolctx = id(node) in self.bool_nodes
        evaluators = []
        variants = []
        where = self._op_columns(left, right, text)
        if where is not None:
            for swap in SWAPS.get(text, ()):
                variants.append(swap)
                evaluators.append(ALWAYS if (text, swap) in COMPLEMENT and id(node) in self.bool_nodes else _binary(FUNCS[swap]))
        # integer constants on either side: the neighbouring values
        consts = []
        for side, operand in (("l", left), ("r", right)):
            if _const_int(operand) and operand.lineno == operand.end_lineno:
                for delta in (1, -1):
                    consts.append((side, operand, delta))
        site = len(SITES)
        op = FUNCS[text]
        site_variants = []
        if where is not None:
            ln, c0, c1 = where
            SITES.append((self.rel, ln, c0, c1, self.lines[ln - 1][c0:c1], list(variants)))
            site_variants = list(evaluators)
        else:
            SITES.append((self.rel, node.lineno, -1, -1, text, []))
        for side, operand, delta in consts:
            SITES.append((self.rel, operand.lineno, operand.col_offset, operand.end_col_offset,
                          str(operand.value), [str(operand.value + delta)]))
        _DETAIL.append((boolctx, op, site_variants))
        for side, operand, delta in consts:
            _DETAIL.append((boolctx, op, [_with_constant(op, side, operand.value + delta)]))
        call = ast.Call(
            func=ast.Name(id="__weak_cmp__", ctx=ast.Load()),
            args=[ast.Constant(site), ast.Constant(len(consts)), left, right],
            keywords=[],
        )
        return ast.copy_location(call, node)

    def visit_UnaryOp(self, node):
        """``not x``: dropping the ``not`` changes the result every time it is evaluated, so
        the site only has to record that it ran (finer than the line: ``a and not b`` may
        stop before ``not b``)."""
        self.generic_visit(node)
        if self.depth == 0 or not isinstance(node.op, ast.Not):
            return node
        line = self.lines[node.lineno - 1]
        if line[node.col_offset : node.col_offset + 4] != "not ":
            return node
        site = len(SITES)
        SITES.append((self.rel, node.lineno, node.col_offset, node.col_offset + 4, "not ", [""]))
        _DETAIL.append((True, None, "not"))
        call = ast.Call(func=ast.Name(id="__weak_not__", ctx=ast.Load()), args=[ast.Constant(site), node.operand], keywords=[])
        return ast.copy_location(call, node)

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if self.depth == 0 or len(node.values) != 2 or id(node) not in self.bool_nodes:
            return node
        a, b = node.values
        if a.end_lineno != b.lineno:
            return node
        line = self.lines[a.end_lineno - 1]
        between = line[a.end_col_offset : b.col_offset]
        word = "and" if isinstance(node.op, ast.And) else "or"
        if between.strip() != word:
            return node
        c0 = a.end_col_offset + between.index(word)
        site = len(SITES)
        SITES.append((self.rel, a.end_lineno, c0, c0 + len(word), word, ["or" if word == "and" else "and"]))
        _DETAIL.append((True, None, word))
        left = ast.Call(func=ast.Name(id="__weak_left__", ctx=ast.Load()), args=[ast.Constant(site), a], keywords=[])
        right = ast.Call(func=ast.Name(id="__weak_right__", ctx=ast.Load()), args=[ast.Constant(site), b], keywords=[])
        node.values = [ast.copy_location(left, a), ast.copy_location(right, b)]
        return node


def _binary(fn):
    return lambda a, b: fn(a, b)


def _with_constant(op, side, value):
    if side == "l":
        return lambda a, b: op(value, b)
    return lambda a, b: op(a, value)


def rewrite(source, rel, filename):
    """Compiles ``source`` with every comparison and boolean swap instrumented."""
    tree = ast.parse(source, filename)
    bool_nodes = _bool_context_nodes(tree)
    tree = _Rewriter(rel, source.split("\n"), bool_nodes).visit(tree)
    ast.fix_missing_locations(tree)
    return compile(tree, filename, "exec", dont_inherit=True)


# ----------------------------------------------------------------------------- recorders
def install(current):
    """Puts the helpers into builtins. ``current()`` returns the set of ``site * 4 + variant``
    codes already seen as differing in the current context."""
    import builtins

    def same(r, m, boolctx):
        if boolctx:
            return bool(r) == bool(m)
        return r is m or (type(r) is type(m) and type(r) in (bool, int, type(None)) and r == m)

    def weak_cmp(site, nconst, a, b):
        _boolctx, op, _evals = _DETAIL[site]
        try:
            result = op(a, b)
        except BaseException as error:
            result, raised = error, True
        else:
            raised = False
        seen = current()
        for offset in range(nconst + 1):
            boolctx, _op, evals = _DETAIL[site + offset]
            for index, evaluate in enumerate(evals):
                code = (site + offset) * 4 + index
                if code in seen:
                    continue
                if evaluate is ALWAYS:
                    seen.add(code)
                    continue
                try:
                    other = evaluate(a, b)
                except BaseException:
                    differs = not raised
                else:
                    try:
                        differs = raised or not same(result, other, boolctx)
                    except BaseException:
                        differs = True
                if differs:
                    seen.add(code)
        if raised:
            raise result
        return result

    def weak_left(site, value):
        # "a and b" -> "a or b": differs whenever "a" is falsy (b is never seen) or "b" is.
        # "a or b" -> "a and b": differs whenever "a" is truthy (b is never seen) or "b" is.
        word = _DETAIL[site][2]
        if bool(value) is (word == "or"):
            current().add(site * 4)
        return value

    def weak_right(site, value):
        word = _DETAIL[site][2]
        if bool(value) is (word == "or"):
            current().add(site * 4)
        return value

    def weak_not(site, value):
        result = not value
        current().add(site * 4)
        return result

    builtins.__weak_cmp__ = weak_cmp
    builtins.__weak_not__ = weak_not
    builtins.__weak_left__ = weak_left
    builtins.__weak_right__ = weak_right
