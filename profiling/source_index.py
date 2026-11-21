"""Per-line facts about one Python source file at the pinned commit.

For every physical line this records:

* ``kind``: what executes it.
    RUNTIME  inside a function body, runs when the function is called
    IMPORT   module or class level, runs once when the module is imported
    MIXED    import-time line that also holds the body of a lambda (``runtime_cols`` says where)
    DOC      a docstring
    NONE     blank lines and comments outside any statement
* ``span``: first and last line of the innermost statement header around the line. An
  ``and``/``or`` or conditional expression can stop a statement before it reaches a later
  line, so the selector charges an edit to everyone who ran any line of the statement.
* ``flags``: string literals and backslash continuations, which decide whether an inserted
  comment line is really inert.
* ``argtypes``: ``arg_types = {...}`` entries on that line, for the arg_types handler.
"""

from __future__ import annotations

import ast
import io
import tokenize

NONE, RUNTIME, IMPORT, MIXED, DOC = 0, 1, 2, 3, 4

IN_STRING = 1  # part of this line is inside a multi-line string literal
STRING_AFTER = 2  # the line break that ends this line is inside a string literal
CONTINUED = 4  # the line ends with a backslash continuation

_STRING_STARTS = {getattr(tokenize, n) for n in ("FSTRING_START", "TSTRING_START") if hasattr(tokenize, n)}
_STRING_ENDS = {getattr(tokenize, n) for n in ("FSTRING_END", "TSTRING_END") if hasattr(tokenize, n)}


def header_span(stmt: ast.stmt) -> tuple[int, int]:
    """Lines evaluated when the statement itself runs, not its nested bodies."""
    end = stmt.end_lineno or stmt.lineno
    if isinstance(stmt, (ast.If, ast.While)):
        end = stmt.test.end_lineno
    elif isinstance(stmt, (ast.For, ast.AsyncFor)):
        end = stmt.iter.end_lineno
    elif isinstance(stmt, (ast.With, ast.AsyncWith)):
        end = max(item.context_expr.end_lineno for item in stmt.items)
    elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        end = stmt.body[0].lineno - 1 if stmt.body else stmt.lineno
        end = max(stmt.lineno, end)
    elif isinstance(stmt, ast.Try) or type(stmt).__name__ == "TryStar":
        end = stmt.lineno
    elif type(stmt).__name__ == "Match":
        end = stmt.subject.end_lineno
    return stmt.lineno, end


def _is_docstring(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


class _Indexer:
    def __init__(self, source: str):
        self.lines = source.split("\n")
        n = len(self.lines) + 2
        self.kind = bytearray(n)
        self.span_start = [0] * n
        self.span_end = [0] * n
        self.depth = [-1] * n
        self.runtime_cols: dict[int, list[tuple[int, int]]] = {}
        self.argtypes: dict[int, list[tuple[str, str]]] = {}
        self.argtypes_classes: dict[str, dict] = {}
        self.module_dicts: dict[int, list[tuple[str, str]]] = {}

    # ------------------------------------------------------------------ marking
    def mark(self, start, end, kind, span, depth):
        for ln in range(start, end + 1):
            if depth >= self.depth[ln]:
                self.depth[ln] = depth
                self.kind[ln] = kind
                self.span_start[ln], self.span_end[ln] = span

    def mark_node(self, node, kind, depth):
        if node is not None:
            self.mark(node.lineno, node.end_lineno, kind, (node.lineno, node.end_lineno), depth)

    def mark_lambdas(self, stmt, depth):
        """Lambda bodies in import-time code run later, when the lambda is called."""
        for sub in ast.walk(stmt):
            if not isinstance(sub, ast.Lambda):
                continue
            body = sub.body
            for ln in range(body.lineno, body.end_lineno + 1):
                c0 = body.col_offset if ln == body.lineno else 0
                c1 = body.end_col_offset if ln == body.end_lineno else 1 << 30
                self.runtime_cols.setdefault(ln, []).append((c0, c1))
                self.kind[ln] = MIXED
                self.span_start[ln], self.span_end[ln] = body.lineno, body.end_lineno

    # ------------------------------------------------------------------ walking
    def visit_body(self, body, scope, qual, depth):
        for i, stmt in enumerate(body):
            if i == 0 and scope != "module-body" and _is_docstring(stmt):
                self.mark(stmt.lineno, stmt.end_lineno, DOC, (stmt.lineno, stmt.end_lineno), depth)
                continue
            if i == 0 and scope == "module-body" and _is_docstring(stmt):
                self.mark(stmt.lineno, stmt.end_lineno, DOC, (stmt.lineno, stmt.end_lineno), depth)
                continue
            self.visit_stmt(stmt, "module" if scope == "module-body" else scope, qual, depth)

    def visit_stmt(self, stmt, scope, qual, depth):
        here = RUNTIME if scope == "func" else IMPORT
        span = header_span(stmt)
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for dec in stmt.decorator_list:
                self.mark_node(dec, here, depth + 1)
            self.mark(span[0], span[1], here, span, depth)
            if isinstance(stmt, ast.ClassDef):
                inner = "func" if scope == "func" else "class"
                self.visit_body(stmt.body, inner, qual + [stmt.name], depth + 1)
                if scope == "class" or scope == "module":
                    self.note_arg_types(stmt, qual + [stmt.name])
            else:
                self.visit_body(stmt.body, "func", qual + [stmt.name, "<locals>"], depth + 1)
            if here == IMPORT:
                self.mark_lambdas_in(stmt.decorator_list, depth)
            return

        # Compound statements keep the scope of their parent for their bodies.
        self.mark(span[0], span[1], here, span, depth)
        if here == IMPORT:
            for part in _header_nodes(stmt):
                self.mark_lambdas(part, depth)
        if scope == "module" and isinstance(stmt, ast.Assign):
            self.note_module_dict(stmt)
        for field in ("body", "orelse", "finalbody"):
            sub = getattr(stmt, field, None)
            if isinstance(sub, list) and sub and isinstance(sub[0], ast.stmt):
                self.visit_body(sub, scope, qual, depth + 1)
        for handler in getattr(stmt, "handlers", []) or []:
            self.mark(handler.lineno, handler.lineno, here, (handler.lineno, handler.lineno), depth + 1)
            if handler.type is not None:
                self.mark_node(handler.type, here, depth + 1)
            self.visit_body(handler.body, scope, qual, depth + 2)
        for case in getattr(stmt, "cases", []) or []:
            self.mark_node(case.pattern, here, depth + 1)
            if case.guard is not None:
                self.mark_node(case.guard, here, depth + 1)
            self.visit_body(case.body, scope, qual, depth + 2)
        # Whatever is left inside the statement (else:, finally:, closing brackets) belongs to it.
        for ln in range(stmt.lineno, (stmt.end_lineno or stmt.lineno) + 1):
            if self.depth[ln] < 0:
                self.depth[ln] = depth
                self.kind[ln] = here
                self.span_start[ln], self.span_end[ln] = span

    def mark_lambdas_in(self, nodes, depth):
        for node in nodes:
            self.mark_lambdas(node, depth)

    # ------------------------------------------------------------------ arg_types
    def note_arg_types(self, cls: ast.ClassDef, qual):
        qualname = ".".join(qual)
        for stmt in cls.body:
            target = None
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
            elif isinstance(stmt, ast.AnnAssign):
                target = stmt.target
            if not (isinstance(target, ast.Name) and target.id == "arg_types"):
                continue
            value = stmt.value
            info = {"line": stmt.lineno, "entries": {}, "spreads": [], "expr": None}
            if isinstance(value, ast.Dict):
                for key, val in zip(value.keys, value.values):
                    if key is None:
                        info["spreads"].append(ast.unparse(val))
                    elif isinstance(key, ast.Constant) and isinstance(key.value, str):
                        info["entries"][key.value] = (val.lineno, val.end_lineno)
                        for ln in range(key.lineno, val.end_lineno + 1):
                            self.argtypes.setdefault(ln, []).append((qualname, key.value))
                    else:
                        info["expr"] = ast.unparse(value)
            else:
                info["expr"] = ast.unparse(value)
            self.argtypes_classes[qualname] = info

    def note_module_dict(self, stmt: ast.Assign):
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            return
        if not isinstance(stmt.value, ast.Dict):
            return
        name = stmt.targets[0].id
        for key, val in zip(stmt.value.keys, stmt.value.values):
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                for ln in range(key.lineno, val.end_lineno + 1):
                    self.module_dicts.setdefault(ln, []).append((name, key.value))


def _header_nodes(stmt):
    if isinstance(stmt, (ast.If, ast.While)):
        return [stmt.test]
    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        return [stmt.target, stmt.iter]
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return [item.context_expr for item in stmt.items]
    if isinstance(stmt, ast.Try) or type(stmt).__name__ == "TryStar":
        return []
    if type(stmt).__name__ == "Match":
        return [stmt.subject]
    return [stmt]


def _token_flags(source: str, lines: list[str]) -> bytearray:
    flags = bytearray(len(lines) + 2)
    comment_at_eol = set()
    starts = []

    def mark_string(srow, erow):
        if erow > srow:
            for r in range(srow, erow + 1):
                flags[r] |= IN_STRING
            for r in range(srow, erow):
                flags[r] |= STRING_AFTER

    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.STRING:
                mark_string(tok.start[0], tok.end[0])
            elif tok.type in _STRING_STARTS:
                starts.append(tok.start[0])
            elif tok.type in _STRING_ENDS and starts:
                mark_string(starts.pop(), tok.end[0])
            elif tok.type == tokenize.COMMENT:
                if tok.end[1] >= len(lines[tok.end[0] - 1].rstrip("\r")):
                    comment_at_eol.add(tok.end[0])
    except (tokenize.TokenError, SyntaxError, IndentationError):
        pass
    for r, text in enumerate(lines, start=1):
        if text.rstrip("\r").endswith("\\") and not flags[r] & STRING_AFTER and r not in comment_at_eol:
            flags[r] |= CONTINUED
    return flags


def index_source(source: str) -> dict:
    """Everything the selector needs to know about one file, as plain data."""
    ix = _Indexer(source)
    tree = ast.parse(source)
    ix.visit_body(tree.body, "module-body", [], 0)
    n = len(ix.lines)
    spans = []
    for ln in range(1, n + 1):
        a, b = ix.span_start[ln], ix.span_end[ln]
        if ix.kind[ln] == NONE or a == 0:
            a = b = ln
        spans.append((a, b))
    return {
        "n": n,
        "kind": bytes(ix.kind[1 : n + 1]),
        "flags": bytes(_token_flags(source, ix.lines)[1 : n + 1]),
        "spans": spans,
        "runtime_cols": ix.runtime_cols,
        "argtypes": ix.argtypes,
        "argtypes_classes": ix.argtypes_classes,
        "module_dicts": ix.module_dicts,
    }
