"""What a patched import actually changed: a fingerprint of every module global and class
attribute of some sqlglot modules, compared between the pinned import (this process) and a
patched one (a forked child).

An edit to code that runs while sqlglot is imported (a class body, a table, a decorator) is hard
to follow by reading it: sqlglot's metaclasses and module code derive many structures from such
values. Measuring is simpler. The fingerprints use tpcds_check.value_digest, which compares
functions by their code (not their line numbers), dicts as mappings and classes by name, and per
key for dicts, so that one changed entry of a table is told apart from the rest.

    fingerprint(names) -> {key: digest}
        ("g", module, name)       a module global
        ("a", owner, attr)        a class attribute, owner = "module.qualname"
        ("k", kind, where, attr)  a dict's digests per key (kind "g" or "a"), {key text: digest}
"""

from __future__ import annotations

import sys
import types

# set by the class machinery or by Python itself; none of them is sqlglot state
SKIP = {
    "__dict__", "__weakref__", "__doc__", "__module__", "__qualname__", "__firstlineno__",
    "__static_attributes__", "__annotations__", "__annotate__", "__orig_bases__", "__parameters__",
    "__abstractmethods__", "_abc_impl", "__slotnames__", "__match_args__",
}


def key_text(key):
    """A stable spelling of a dict key; the profile's spelling (trace_plugin._key_text)."""
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


def classes_of(module):
    """Classes defined in a module, nested ones included."""
    out = []
    stack = [v for v in vars(module).values() if isinstance(v, type) and v.__module__ == module.__name__]
    seen = set()
    while stack:
        cls = stack.pop()
        if id(cls) in seen:
            continue
        seen.add(id(cls))
        out.append(cls)
        for v in vars(cls).values():
            if isinstance(v, type) and v.__module__ == module.__name__ and v.__qualname__.startswith(cls.__qualname__ + "."):
                stack.append(v)
    return out


def fingerprint(names, digest):
    """``digest(value, memo)``; one memo serves the whole run (shared objects are digested once)."""
    memo = {}
    keep = []  # the memo is by id: what it saw must stay alive until the end
    plain = digest
    digest = lambda v: (keep.append(v), plain(v, memo))[1]
    out = {}
    for name in names:
        module = sys.modules.get(name)
        if module is None:
            continue
        _entries(out, "g", name, vars(module), digest)
        for cls in classes_of(module):
            _entries(out, "a", "%s.%s" % (name, cls.__qualname__), vars(cls), digest)
    return out


def _entries(out, kind, where, namespace, digest):
    for attr, value in list(namespace.items()):
        if attr in SKIP or (kind == "g" and attr.startswith("__") and attr.endswith("__")):
            continue
        if kind == "g" and isinstance(value, types.ModuleType):
            continue  # a submodule bound on its package
        out[(kind, where, attr)] = digest(value)
        if isinstance(value, dict) and len(value) > 1:
            keys = {}
            for k, v in list(value.items()):
                text = key_text(k)
                if text is None:
                    keys = None
                    break
                keys[text] = digest(v) + "|" + text
            if keys is not None:
                out[("k", kind, where, attr)] = keys


def differences(before, after):
    """{key: "changed" | "added" | "removed"} for module globals and class attributes, and for
    dicts compared by key, ("k", ...) -> set of the key texts that differ."""
    out = {}
    for key in set(before) | set(after):
        if key[0] == "k":
            continue
        if key not in after:
            out[key] = "removed"
        elif key not in before:
            out[key] = "added"
        elif before[key] != after[key]:
            out[key] = "changed"
            kk = ("k",) + key
            a, b = before.get(kk), after.get(kk)
            if a is not None and b is not None:
                out[kk] = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    return out
