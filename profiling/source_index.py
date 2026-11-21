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
        # line -> [(class qualname, attribute)] for class-level assignments ("" for module level)
        self.classattr: dict[int, list[tuple[str, str]]] = {}
        # line -> [(col0, col1, body_start, body_end)] for default values of functions
        self.defaults: dict[int, list[tuple[int, int, int, int]]] = {}
        # (scope, names defined, names read) for import-time statements and class-creation hooks
        self.derivations: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
        # identifier -> lines of runtime statements that mention it
        self.runtime_names: dict[str, set[int]] = {}
        self.hook_owner = ""
        self.pure_lines: set[int] = set()
        # line -> [(class qualname, attribute, key source, value col0, value col1)]
        self.dict_entries: dict[int, list[tuple]] = {}
        self.dict_displays: list[tuple[str, str]] = []
        # line -> column ranges of annotations that are never evaluated
        self.annotations: dict[int, list[tuple[int, int]]] = {}
        self.postponed = False
        self.decorated_class = False
        # modules imported while this module is being imported (module and class level)
        self.imports: set[tuple[int, str, tuple[str, ...]]] = set()

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

    def note_annotation(self, node):
        for ln in range(node.lineno, node.end_lineno + 1):
            c0 = node.col_offset if ln == node.lineno else 0
            c1 = node.end_col_offset if ln == node.end_lineno else 1 << 30
            self.annotations.setdefault(ln, []).append((c0, c1))

    def visit_stmt(self, stmt, scope, qual, depth):
        here = RUNTIME if scope == "func" else IMPORT
        span = header_span(stmt)
        # Annotations that nothing evaluates: local variable annotations always; with
        # "from __future__ import annotations", those of undecorated functions and of plain
        # (undecorated, so not dataclass) class bodies too.
        if isinstance(stmt, ast.AnnAssign):
            if scope == "func" or (self.postponed and not (scope == "class" and self.decorated_class)):
                self.note_annotation(stmt.annotation)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and self.postponed and all(
            _plain_decorator(d) for d in stmt.decorator_list
        ):
            args = stmt.args
            for arg in args.posonlyargs + args.args + args.kwonlyargs + [args.vararg, args.kwarg]:
                if arg is not None and arg.annotation is not None:
                    self.note_annotation(arg.annotation)
            if stmt.returns is not None:
                self.note_annotation(stmt.returns)
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for dec in stmt.decorator_list:
                self.mark_node(dec, here, depth + 1)
            self.mark(span[0], span[1], here, span, depth)
            if here == IMPORT and not isinstance(stmt, ast.ClassDef):
                self.note_defaults(stmt)
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name in _HOOKS:
                self.hook_owner = ".".join(qual)
                self.note_hook(stmt)
            if isinstance(stmt, ast.ClassDef):
                inner = "func" if scope == "func" else "class"
                outer, self.decorated_class = self.decorated_class, bool(stmt.decorator_list)
                self.visit_body(stmt.body, inner, qual + [stmt.name], depth + 1)
                self.decorated_class = outer
                if scope == "class" or scope == "module":
                    self.note_arg_types(stmt, qual + [stmt.name])
            else:
                self.visit_body(stmt.body, "func", qual + [stmt.name, "<locals>"], depth + 1)
            if here == IMPORT:
                self.mark_lambdas_in(stmt.decorator_list, depth)
            return

        if here == IMPORT and isinstance(stmt, ast.Import):
            for alias in stmt.names:
                self.imports.add((0, alias.name, ()))
        elif here == IMPORT and isinstance(stmt, ast.ImportFrom):
            self.imports.add((stmt.level, stmt.module or "", tuple(a.name for a in stmt.names)))
        # Compound statements keep the scope of their parent for their bodies.
        self.mark(span[0], span[1], here, span, depth)
        if here == RUNTIME:
            for part in _header_nodes(stmt):
                for name in _names_read(part):
                    self.runtime_names.setdefault(name, set()).update(range(span[0], span[1] + 1))
        if here == IMPORT:
            for part in _header_nodes(stmt):
                self.mark_lambdas(part, depth)
            self.note_derivation(stmt, scope, ".".join(qual))
            if scope in ("class", "module"):
                self.note_classattr(stmt, ".".join(qual))
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

    # ------------------------------------------------------------------ class attributes
    def note_dict_entries(self, stmt, qualname, names):
        """Entries of a dict display assigned at class or module level, so the selector can
        charge an edit to one entry's value to the tests that looked up that key."""
        value = getattr(stmt, "value", None)
        if not isinstance(value, ast.Dict) or len(names) != 1:
            return
        self.dict_displays.append((qualname, names[0]))
        for key, val in zip(value.keys, value.values):
            if key is None or key.lineno != val.end_lineno or key.lineno != key.end_lineno:
                continue
            if not (isinstance(key, ast.Constant) or _dotted(key)):
                continue
            self.dict_entries.setdefault(key.lineno, []).append(
                (qualname, names[0], ast.unparse(key), val.col_offset, val.end_col_offset)
            )

    def note_classattr(self, stmt, qualname):
        names = _target_names(stmt)
        if not names or not isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            return
        if not isinstance(stmt, ast.AugAssign):
            self.note_dict_entries(stmt, qualname, names)
        pure = isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is not None and _is_pure(stmt.value)
        for ln in range(stmt.lineno, (stmt.end_lineno or stmt.lineno) + 1):
            for name in names:
                self.classattr.setdefault(ln, []).append((qualname, name))
            if pure:
                self.pure_lines.add(ln)

    def note_derivation(self, stmt, scope, qualname):
        """Import-time statement: which names it (re)defines from which names it reads.

        Reads are kept apart: bare names (class-body names or module globals) and attribute
        references with their receiver spelled out (``Parser.FUNCTIONS`` -> ("Parser",
        "FUNCTIONS")), so the map builder can resolve the receiver to the class that owns it."""
        targets = _target_names(stmt)
        if not targets:
            return
        value = getattr(stmt, "value", None)
        bare, refs = _reads_split(value) if value is not None else (set(), set())
        if isinstance(stmt, ast.AugAssign):
            bare |= set(targets)
        if bare or refs:
            keyonly = tuple(sorted(_key_only_names(value))) if value is not None else ()
            self.derivations.append(
                (scope, qualname, tuple(sorted(targets)), tuple(sorted(bare)), tuple(sorted(refs)), keyonly)
            )

    def note_hook(self, func):
        """Class-creation hooks (__init_subclass__, metaclass __new__/__init__) derive attributes,
        e.g. ``cls._KEYWORD_TRIE = new_trie(... cls.KEYWORDS ...)`` or
        ``klass.ESCAPED_SEQUENCES.setdefault(...)``. Every attribute such a hook assigns or
        mutates becomes a derivation from the names its value reads, including local variables
        (transitively) and the conditions the statement sits under."""
        local_reads: dict[str, set] = {}
        derived = []

        def plain_names(target):
            if isinstance(target, ast.Name):
                return [target.id]
            if isinstance(target, (ast.Tuple, ast.List)):
                return [n for elt in target.elts for n in plain_names(elt)]
            if isinstance(target, ast.Starred):
                return plain_names(target.value)
            return []

        def walk(body, guards):
            for node in body:
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and getattr(node, "value", None) is not None:
                    reads = _names_read(node.value) | guards
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    attrs = set()
                    for target in targets:
                        attrs |= {sub.attr for sub in ast.walk(target) if isinstance(sub, ast.Attribute)}
                        for name in plain_names(target):
                            local_reads.setdefault(name, set()).update(reads)
                    if attrs:
                        derived.append((attrs, reads))
                elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    fn = node.value.func
                    if isinstance(fn, ast.Attribute) and isinstance(fn.value, (ast.Attribute, ast.Subscript)):
                        attrs = {sub.attr for sub in ast.walk(fn.value) if isinstance(sub, ast.Attribute)}
                        if attrs:
                            derived.append((attrs, _names_read(node.value) | guards))
                inner = set(guards)
                if isinstance(node, (ast.If, ast.While)):
                    inner |= _names_read(node.test)
                elif isinstance(node, (ast.For, ast.AsyncFor)):
                    inner |= _names_read(node.iter)
                    for name in plain_names(node.target):
                        local_reads.setdefault(name, set()).update(_names_read(node.iter) | guards)
                elif isinstance(node, (ast.With, ast.AsyncWith)):
                    for item in node.items:
                        if item.optional_vars is not None:
                            for name in plain_names(item.optional_vars):
                                local_reads.setdefault(name, set()).update(_names_read(item.context_expr))
                for field in ("body", "orelse", "finalbody"):
                    sub = getattr(node, field, None)
                    if isinstance(sub, list) and sub and isinstance(sub[0], ast.stmt):
                        walk(sub, inner)
                for handler in getattr(node, "handlers", []) or []:
                    walk(handler.body, inner)

        walk(func.body, set())
        keyonly = tuple(sorted(_key_only_names(func)))
        for attrs, reads in derived:
            reads = set(reads)
            frontier = list(reads)
            while frontier:
                for extra in local_reads.get(frontier.pop(), ()):
                    if extra not in reads:
                        reads.add(extra)
                        frontier.append(extra)
            self.derivations.append(("hook", self.hook_owner, tuple(sorted(attrs)), tuple(sorted(reads)), (), keyonly))

    def note_defaults(self, func):
        args = func.args
        values = list(args.defaults) + [d for d in args.kw_defaults if d is not None]
        if not values or not func.body:
            return
        b0 = func.body[0].lineno
        b1 = func.end_lineno or b0
        for value in values:
            for ln in range(value.lineno, value.end_lineno + 1):
                c0 = value.col_offset if ln == value.lineno else 0
                c1 = value.end_col_offset if ln == value.end_lineno else 1 << 30
                self.defaults.setdefault(ln, []).append((c0, c1, b0, b1))

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


_HOOKS = {"__init_subclass__", "__new__", "__init__", "__set_name__"}
# Decorators that never look at a function's annotations.
_PLAIN_DECORATORS = {"classmethod", "staticmethod", "property", "abstractmethod", "cached_property", "override"}


def _plain_decorator(node):
    if isinstance(node, ast.Name):
        return node.id in _PLAIN_DECORATORS
    if isinstance(node, ast.Attribute):
        return node.attr in _PLAIN_DECORATORS or node.attr in ("setter", "getter", "deleter")
    return False


def _target_names(stmt):
    if isinstance(stmt, ast.Assign):
        targets = stmt.targets
    elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
        targets = [stmt.target]
    else:
        return []
    names = []
    for target in targets:
        for sub in ast.walk(target):
            if isinstance(sub, ast.Name):
                names.append(sub.id)
    return names


_PURE_NODES = (
    ast.Constant, ast.Name, ast.Attribute, ast.Dict, ast.List, ast.Tuple, ast.Set, ast.Load,
    ast.UnaryOp, ast.USub, ast.UAdd, ast.Not, ast.Starred,
)


def _is_pure(node):
    """A literal display of constants and names: evaluating it cannot raise or run code
    (attribute lookups on sqlglot classes and modules are plain reads)."""
    return all(isinstance(sub, _PURE_NODES) for sub in ast.walk(node))


_KEY_CALLS = {"len", "sorted", "set", "list", "frozenset", "iter", "tuple"}


def _key_only_names(root):
    """Names that ``root`` only ever uses by key: as the right side of ``in``/``not in``, as
    something iterated, through ``.keys()``, or in len()/sorted()/set()/list(). A change to a
    dict entry's value cannot reach anything derived through such uses. A name assigned to a
    local variable counts as used by key when every use of that local is."""
    parents = {}
    for node in ast.walk(root):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def key_use(node):
        parent = parents.get(node)
        if isinstance(parent, ast.Compare) and len(parent.ops) == 1 and isinstance(parent.ops[0], (ast.In, ast.NotIn)):
            return parent.comparators[0] is node
        if isinstance(parent, (ast.comprehension, ast.For, ast.AsyncFor)):
            return parent.iter is node
        if isinstance(parent, ast.Attribute) and parent.attr == "keys":
            return isinstance(parents.get(parent), ast.Call)
        if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name) and parent.func.id in _KEY_CALLS:
            return node in parent.args
        return False

    local_uses = {}
    for node in ast.walk(root):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            local_uses.setdefault(node.id, []).append(key_use(node))

    uses = {}
    for node in ast.walk(root):
        if isinstance(node, ast.Attribute):
            name = node.attr
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            name = node.id
        else:
            continue
        parent = parents.get(node)
        if (isinstance(parent, ast.Assign) and parent.value is node and len(parent.targets) == 1
                and isinstance(parent.targets[0], ast.Name)):
            alias = local_uses.get(parent.targets[0].id, [])
            uses.setdefault(name, []).append(bool(alias) and all(alias))
        else:
            uses.setdefault(name, []).append(key_use(node))
    return {name for name, flags in uses.items() if flags and all(flags)}


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _reads_split(node):
    """(bare names, {(receiver or "", attribute)}) mentioned by an expression."""
    bare, refs = set(), set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            bare.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            refs.add((_dotted(sub.value) or "", sub.attr))
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str) and sub.value.isidentifier():
            refs.add(("", sub.value))  # getattr(obj, "NAME")
    return bare, refs


def _names_read(node):
    """Every bare name and attribute name an expression mentions."""
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            out.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            out.add(sub.attr)
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str) and sub.value.isidentifier():
            out.add(sub.value)  # getattr(cls, "NAME")
    return out


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


def compile_units(tree) -> list[tuple[int, int]]:
    """Line ranges that compile on their own once dedented: module-level statements, and inside
    classes each statement of the class body (methods with their decorators). An edit inside
    one of them can be syntax-checked by compiling just that range."""
    units = []

    def first_line(stmt):
        decorators = getattr(stmt, "decorator_list", None) or []
        return min([stmt.lineno] + [d.lineno for d in decorators])

    def walk(body):
        for stmt in body:
            if isinstance(stmt, ast.ClassDef):
                walk(stmt.body)
            else:
                units.append((first_line(stmt), stmt.end_lineno or stmt.lineno))

    walk(tree.body)
    return sorted(units)


def class_lines(tree) -> dict[int, str]:
    """line -> qualified name of the innermost class whose body (methods included) holds it."""
    out = {}

    def walk(node, qual):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                name = qual + [child.name]
                for ln in range(child.lineno, (child.end_lineno or child.lineno) + 1):
                    out[ln] = ".".join(name)
                walk(child, name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                walk(child, qual + [child.name, "<locals>"])
            else:
                walk(child, qual)

    walk(tree, [])
    return out


def all_imports(tree) -> list[tuple[int, str, tuple[str, ...]]]:
    """Every import statement, wherever it sits (tests import dialects inside test methods)."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add((0, alias.name, ()))
        elif isinstance(node, ast.ImportFrom):
            out.add((node.level, node.module or "", tuple(a.name for a in node.names)))
    return sorted(out)


def lazy_imports(tree) -> list[tuple[int, int, str, tuple[str, ...]]]:
    """Import statements inside function bodies: (line, level, module, names). They run when
    the function does, so a module that no longer imports breaks whoever calls them."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Import):
                    for alias in sub.names:
                        out.add((sub.lineno, 0, alias.name, ()))
                elif isinstance(sub, ast.ImportFrom):
                    out.add((sub.lineno, sub.level, sub.module or "", tuple(a.name for a in sub.names)))
    return sorted(out)


def index_source(source: str) -> dict:
    """Everything the selector needs to know about one file, as plain data."""
    ix = _Indexer(source)
    tree = ast.parse(source)
    ix.postponed = any(
        isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__" and any(a.name == "annotations" for a in stmt.names)
        for stmt in tree.body
    )
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
        "classattr": ix.classattr,
        "defaults": ix.defaults,
        "derivations": ix.derivations,
        "runtime_names": {k: sorted(v) for k, v in ix.runtime_names.items()},
        "imports": sorted(ix.imports),
        "pure": sorted(ix.pure_lines),
        "units": compile_units(tree),
        "annotations": ix.annotations,
        "dict_entries": ix.dict_entries,
        "class_of": class_lines(tree),
        "all_imports": all_imports(tree),
        "lazy_imports": lazy_imports(tree),
        "dict_displays": ix.dict_displays,
    }
