"""Which tests can observe an edit to a value that is computed at import time.

An edit to ``TSQL.WEEK_OFFSET = -1`` or to an entry of ``SingleStoreParser.SHOW_PARSERS``
changes one class attribute. Its effect reaches a test only if the test reads that attribute,
or reads something derived from it while the classes were being built:

* class bodies that copy it (``FUNCTIONS = {**Parser.FUNCTIONS, ...}``,
  ``SHOW_TRIE = new_trie(... SHOW_PARSERS)``), resolved to the class that actually owns the
  attribute read, so ``Parser.FUNCTIONS`` is not confused with ``DuckDBParser.FUNCTIONS``;
* class-creation hooks (``__init_subclass__``, metaclass ``__new__``) that compute attributes
  of every subclass and of a dialect's tokenizer, parser and generator;
* module globals built from it, and the bindings other modules import them under
  (``from sqlglot.typing.spark import EXPRESSION_METADATA``), each owned by its module: the
  readers of ``sqlglot.typing.mysql``'s table are not those of ``sqlglot.typing.duckdb``'s.

The profile says which tests read which ``owner:attribute`` (owner = the class whose ``__dict__``
holds it, so inherited reads are charged to the defining class). This module computes the
closure of affected (owner, attribute) pairs and turns it into a test bitset.
"""

from __future__ import annotations

import builtins
import sys
from collections import defaultdict

GLOBAL = -2
ANY = "*"
MODULE = "<module>"  # owner of a module's globals: "<module>sqlglot.typing.mysql"


def mkey(modname):
    return MODULE + modname


def is_module(owner):
    return owner.startswith(MODULE)


def absolute(modname, is_package, level, source):
    """The module a "from <level dots><source> import ..." in module ``modname`` names."""
    if not level:
        return source
    parts = modname.split(".") if is_package else modname.split(".")[:-1]
    if level > 1:
        parts = parts[: len(parts) - (level - 1)] if len(parts) >= level - 1 else []
    if not parts:
        return None
    return ".".join(parts + ([source] if source else []))


def key_text(key):
    """A stable spelling of a dict key. Must match trace_plugin._key_text."""
    import enum

    if isinstance(key, bool) or key is None:
        return "k:%r" % (key,)
    if isinstance(key, (str, int, float)):
        return "%s:%r" % (type(key).__name__, key)
    if isinstance(key, enum.Enum):
        return "e:%s.%s.%s" % (type(key).__module__, type(key).__qualname__, key.name)
    if isinstance(key, type):
        return "c:%s.%s" % (key.__module__, key.__qualname__)
    if isinstance(key, tuple):
        return "t:(%s)" % ",".join(key_text(k) for k in key)
    return None


def module_name(rel):
    name = rel[:-3].replace("/", ".")
    return name[: -len(".__init__")] if name.endswith(".__init__") else name


class ImportEffects:
    def __init__(self, files, families, instrumented, reads, global_reads, name_cov):
        """files: rel -> source index; families: class key -> (mro keys, component keys);
        instrumented: {(owner, attr)}; reads: "owner:attr" -> bits; global_reads: tags read
        in contexts charged to every test; name_cov(module, name) -> bits or GLOBAL, the tests
        that read that module global."""
        self.families = families
        self.instrumented = instrumented
        self.reads = reads
        self.global_reads = global_reads
        self.name_cov = name_cov
        self._family = {}
        self._objects = {}
        self._subclasses = defaultdict(set)
        for key, (mro, _parts) in families.items():
            for base in mro:
                self._subclasses[base].add(key)
        self.derivations = []  # (scope, owner key, targets, resolved reads, hook names)
        self.by_name = defaultdict(list)
        self.star = defaultdict(set)  # module key -> keys of the modules that import * from it
        for rel, ix in files.items():
            if not rel.startswith("sqlglot/"):
                continue
            modname = module_name(rel)
            here = mkey(modname)
            mod = sys.modules.get(modname)
            for level, source, name, alias in ix.get("import_binds", ()):
                origin = absolute(modname, rel.endswith("/__init__.py"), level, source)
                if origin is None:
                    continue
                if name == "*":
                    self.star[mkey(origin)].add(here)
                    continue
                # the import binds this module's ``alias`` to the other module's ``name``
                self.by_name[name].append(len(self.derivations))
                self.derivations.append(("module", here, (alias,), frozenset({(mkey(origin), name)}), frozenset(), frozenset()))
            for scope, qualname, targets, bare, refs, *rest in ix["derivations"]:
                keyonly = frozenset(rest[0]) if rest else frozenset()
                owner = "%s.%s" % (modname, qualname) if qualname else here
                if scope == "hook":
                    entry = (scope, owner, targets, (), frozenset(bare), keyonly)
                    names = set(bare)
                else:
                    resolved = set()
                    for name in bare:
                        resolved.add((here, name))
                        if scope == "class":
                            resolved.add((owner, name))
                    for receiver, attr in refs:
                        resolved |= self.resolve(mod, owner if scope == "class" else None, receiver, attr)
                    entry = (scope, owner, targets, frozenset(resolved), frozenset(), keyonly)
                    names = {n for _o, n in resolved}
                index = len(self.derivations)
                self.derivations.append(entry)
                for name in names:
                    self.by_name[name].append(index)
        self._cache = {}

    # ------------------------------------------------------------------ static resolution
    def resolve(self, mod, owner, receiver, attr):
        """The (owner class, attribute) an ``receiver.attr`` reference in import-time code reads."""
        if mod is None or not receiver:
            return {(ANY, attr)}
        parts = receiver.split(".")
        obj = None
        if owner is not None:
            cls = self._class(owner)
            if cls is not None and parts[0] in vars(cls):
                obj = vars(cls)[parts[0]]
        if obj is None:
            obj = vars(mod).get(parts[0], getattr(builtins, parts[0], None))
        for part in parts[1:]:
            obj = getattr(obj, part, None)
        if isinstance(obj, type):
            for klass in obj.__mro__:
                if attr in vars(klass):
                    return {("%s.%s" % (klass.__module__, klass.__qualname__), attr)}
            return {(ANY, attr)}
        if isinstance(obj, type(sys)):
            return {(mkey(obj.__name__), attr)}
        return {(ANY, attr)}

    def _class(self, key):
        """The class object for a "module.qualname" key, if it can be found."""
        if key in self._objects:
            return self._objects[key]
        parts = key.split(".")
        self._objects[key] = None
        for i in range(len(parts) - 1, 0, -1):
            mod = sys.modules.get(".".join(parts[:i]))
            if mod is not None:
                obj = mod
                for part in parts[i:]:
                    obj = getattr(obj, part, None)
                self._objects[key] = obj if isinstance(obj, type) else None
                break
        return self._objects[key]

    # ------------------------------------------------------------------ families
    def family(self, key):
        """The class, its subclasses, dialects built from any of them, and those dialects'
        own subclasses and components: everything a class-creation hook can derive into."""
        fam = self._family.get(key)
        if fam is not None:
            return fam
        fam = set(self._subclasses.get(key, ())) | {key}
        dialects = {d for d, (_mro, parts) in self.families.items() if any(p in fam for p in parts)}
        for d in dialects:
            fam |= self._subclasses.get(d, set()) | {d}
        for d in list(fam):
            for part in self.families.get(d, ((), ()))[1]:
                fam |= self._subclasses.get(part, set()) | {part}
        self._family[key] = fam
        return fam

    def hook_family(self, hook_owner):
        """Every class a hook runs for: subclasses of the class defining ``__init_subclass__``,
        or the classes built by a metaclass."""
        fam = self._family.get(("hook", hook_owner))
        if fam is not None:
            return fam
        holder = self._class(hook_owner)
        fam = set()
        if holder is None:
            fam = set(self.families)
        else:
            for key in self.families:
                cls = self._class(key)
                if cls is None:
                    continue
                if (issubclass(holder, type) and isinstance(cls, holder)) or (
                    not issubclass(holder, type) and issubclass(cls, holder)
                ):
                    fam |= self._subclasses.get(key, set()) | {key}
        self._family[("hook", hook_owner)] = fam
        return fam

    # ------------------------------------------------------------------ closure
    def affected(self, owner, attr, values_only=False):
        """Every (owner, attribute) whose value an edit to ``owner.attr`` can change. With
        ``values_only`` the edit changes only the value of an existing dict entry, so code that
        uses ``attr`` purely by key (membership tests, iteration over keys) is not affected."""
        start = (owner, attr)
        done = {start}
        names = {attr}
        frontier = [start]
        while frontier:
            o, n = frontier.pop()
            if is_module(o) and not n.startswith("_"):
                for importer in self.star.get(o, ()):
                    if (importer, n) not in done:
                        done.add((importer, n))
                        frontier.append((importer, n))
            for index in self.by_name.get(n, ()):
                scope, d_owner, targets, reads, hook_names, keyonly = self.derivations[index]
                if values_only and (o, n) == start and n in keyonly:
                    continue
                new = []
                if scope == "hook":
                    members = self.hook_family(d_owner) if o == ANY or is_module(o) else self.family(o)
                    for member in members:
                        new.extend((member, t) for t in targets)
                elif (o, n) in reads or (ANY, n) in reads:
                    new.extend((d_owner, t) for t in targets)
                for pair in new:
                    if pair not in done:
                        done.add(pair)
                        names.add(pair[1])
                        frontier.append(pair)
        return done

    def resolve_key(self, rel, qualname, source):
        """The runtime spelling of a dict entry's key written as ``source`` in that class."""
        import ast as _ast

        try:
            return key_text(_ast.literal_eval(source))
        except (ValueError, SyntaxError):
            pass
        mod = sys.modules.get(module_name(rel))
        if mod is None:
            return None
        parts = source.split(".")
        obj = None
        cls = self._class("%s.%s" % (module_name(rel), qualname)) if qualname else None
        if cls is not None and parts[0] in vars(cls):
            obj = vars(cls)[parts[0]]
        if obj is None:
            obj = vars(mod).get(parts[0], getattr(builtins, parts[0], None))
        for part in parts[1:]:
            obj = getattr(obj, part, None)
        return key_text(obj) if obj is not None else None

    def tests_for_key(self, owner, attr, key, keyed, displays):
        """Like tests_for, for an edit to the value of one entry of a dict attribute: the dict
        itself and copies made by spreading it into another dict display are charged by key,
        everything derived in any other way by attribute."""
        if not is_module(owner) and (owner not in self.families or (owner, attr) not in self.instrumented):
            return GLOBAL
        bits = 0
        for o, n in self.affected(owner, attr, values_only=True):
            if is_module(o):
                cov = self.name_cov(o[len(MODULE):], n)
                if cov == GLOBAL:
                    return GLOBAL
                bits |= cov
                continue
            if o == ANY:
                return GLOBAL
            tag = "%s:%s" % (o, n)
            if tag in self.global_reads:
                return GLOBAL
            if n == attr and (o == owner or (o, n) in displays):
                got = keyed(tag, key)
            else:
                got = self.reads.get(tag, 0)
            if got == GLOBAL:
                return GLOBAL
            bits |= got
        return bits

    def tests_for(self, owner, attr):
        """Bitset of tests that can observe a change to ``owner.attr``, or GLOBAL."""
        key = (owner, attr)
        if key in self._cache:
            return self._cache[key]
        result = self._compute(owner, attr)
        self._cache[key] = result
        return result

    def _compute(self, owner, attr):
        if not is_module(owner):
            if owner not in self.families or (owner, attr) not in self.instrumented:
                return GLOBAL
        bits = 0
        for o, n in self.affected(owner, attr):
            if is_module(o):
                cov = self.name_cov(o[len(MODULE):], n)
                if cov == GLOBAL:
                    return GLOBAL
                bits |= cov
                continue
            if o == ANY:
                return GLOBAL
            tag = "%s:%s" % (o, n)
            if tag in self.global_reads:
                return GLOBAL
            bits |= self.reads.get(tag, 0)
        return bits
