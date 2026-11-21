"""Which tests can observe an edit to a value that is computed at import time.

An edit to ``TSQL.WEEK_OFFSET = -1`` or to an entry of ``SingleStoreParser.SHOW_PARSERS``
changes one class attribute. Its effect reaches a test only if the test reads that attribute,
or reads something derived from it while the classes were being built:

* class bodies that copy it (``FUNCTIONS = {**Parser.FUNCTIONS, ...}``,
  ``SHOW_TRIE = new_trie(... SHOW_PARSERS)``), resolved to the class that actually owns the
  attribute read, so ``Parser.FUNCTIONS`` is not confused with ``DuckDBParser.FUNCTIONS``;
* class-creation hooks (``__init_subclass__``, metaclass ``__new__``) that compute attributes
  of every subclass and of a dialect's tokenizer, parser and generator;
* module globals built from it, whose readers are the statements that mention them.

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
MODULE = "<global>"


def module_name(rel):
    name = rel[:-3].replace("/", ".")
    return name[: -len(".__init__")] if name.endswith(".__init__") else name


class ImportEffects:
    def __init__(self, files, families, instrumented, reads, global_reads, name_cov):
        """files: rel -> source index; families: class key -> (mro keys, component keys);
        instrumented: {(owner, attr)}; reads: "owner:attr" -> bits; global_reads: tags read
        in contexts charged to every test; name_cov(name) -> bits or GLOBAL."""
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
        for rel, ix in files.items():
            if not rel.startswith("sqlglot/"):
                continue
            mod = sys.modules.get(module_name(rel))
            for scope, qualname, targets, bare, refs in ix["derivations"]:
                owner = "%s.%s" % (module_name(rel), qualname) if qualname else MODULE
                if scope == "hook":
                    entry = (scope, owner, targets, (), frozenset(bare))
                    names = set(bare)
                else:
                    resolved = set()
                    for name in bare:
                        resolved.add((MODULE, name))
                        if scope == "class":
                            resolved.add((owner, name))
                    for receiver, attr in refs:
                        resolved |= self.resolve(mod, owner if scope == "class" else None, receiver, attr)
                    entry = (scope, owner, targets, frozenset(resolved), frozenset())
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
            return {(MODULE, attr)}
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
    def affected(self, owner, attr):
        start = (owner, attr)
        done = {start}
        names = {attr}
        frontier = [start]
        while frontier:
            o, n = frontier.pop()
            for index in self.by_name.get(n, ()):
                scope, d_owner, targets, reads, hook_names = self.derivations[index]
                new = []
                if scope == "hook":
                    members = self.hook_family(d_owner) if o in (MODULE, ANY) else self.family(o)
                    for member in members:
                        new.extend((member, t) for t in targets)
                elif (o, n) in reads or (ANY, n) in reads:
                    if scope == "class":
                        new.extend((d_owner, t) for t in targets)
                    else:
                        new.extend((MODULE, t) for t in targets)
                for pair in new:
                    if pair not in done:
                        done.add(pair)
                        names.add(pair[1])
                        frontier.append(pair)
        return done

    def tests_for(self, owner, attr):
        """Bitset of tests that can observe a change to ``owner.attr``, or GLOBAL."""
        key = (owner, attr)
        if key in self._cache:
            return self._cache[key]
        result = self._compute(owner, attr)
        self._cache[key] = result
        return result

    def _compute(self, owner, attr):
        if owner != MODULE:
            if owner not in self.families or (owner, attr) not in self.instrumented:
                return GLOBAL
        bits = 0
        for o, n in self.affected(owner, attr):
            if o == MODULE:
                cov = self.name_cov(n)
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
