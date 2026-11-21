"""pytest plugin that records what every sqlglot test touches.

Loaded with ``-p trace_plugin`` while running the sqlglot suite at the pinned commit.
It writes one gzip text file (``PROFILE_OUT``) with these records:

    F <fid>\t<path>               file id table, paths relative to the source root
    C <ctx>\t<fid>:<line>,...     lines executed in a context (negative: module or class body)
    O <ctx>\t<path>|<path>|...    files opened, or directories listed (trailing "/")
    V <ctx>\t<class>:<arg>,...    expression args that were missing when the node was validated
    R <ctx>\t<class>:<attr>,...   class attributes read (with PROFILE_READS=1)
    A <class>\t<attr>,...         attributes that were instrumented
    K <class>\t<mro>\t<parts>     class family: MRO and a dialect's component classes
    I <fid>:<line>\t<fid>:<line>,...  import-time statements a function line ran on behalf of
    Y <ctx>\t[[<class>:<attr>, <key>], ...]  dict keys looked up (PROFILE_KEYS=1; "*" = all)

Contexts:

    <import>                      importing every sqlglot module before collection
    <collect>|<module>            importing a test module during collection
    <fixture>|<scope>|<node>      setting up a class, module or session scoped fixture
    <nodeid>|setup / |call / |teardown
    <other>                       everything else

Lines are collected with ``sys.monitoring`` (PEP 669). Each (code, line) location reports once
and is then disabled; ``restart_events()`` re-arms everything whenever the context changes, so
the cost grows with the number of distinct lines a test runs, not with how often it runs them.

Two details matter for getting the attribution right:

* sqlglot imports dialects lazily. If the first test that needs a dialect also imported it,
  every line that ran at import time would be charged to that one test. Importing the whole
  package up front puts all import-time work in ``<import>``, where the selector treats it as
  global.
* ``test_optimizer`` and ``test_executor`` run their cases in process pools. Lines executed in
  a worker process are invisible here, so pools are replaced by an executor that runs each task
  in this process. Arguments and results still go through pickle, like they would across a
  process boundary, because pickling expressions runs sqlglot code too.
"""

import gzip
import os
import sys

import pytest

mon = sys.monitoring
TOOL = mon.COVERAGE_ID
CO_OPTIMIZED = 0x0001
ROOT = os.path.abspath(os.environ.get("PROFILE_ROOT", os.getcwd())) + os.sep
OUT = os.environ.get("PROFILE_OUT", "profile.txt.gz")

_files = {}
_ctx = "<other>"
_lines = {}
_reads = {}
_missing = {}
_origins = {}
_out = None


def _fid(path):
    fid = _files.get(path)
    if fid is None:
        fid = _files[path] = len(_files)
    return fid


def _on_line(code, line):
    path = code.co_filename
    if path.startswith(ROOT):
        hits = _lines.get(_ctx)
        if hits is None:
            hits = _lines[_ctx] = set()
        # Module and class bodies are not CO_OPTIMIZED; their lines are stored negated so a
        # lambda created at import time can be told apart from the same lambda being called.
        hits.add((_fid(path), line if code.co_flags & CO_OPTIMIZED else -line))
        if _ctx == "<import>":
            _note_origin(code, path, line)
    return mon.DISABLE


def _note_origin(code, path, line):
    """While importing, remember which module or class body statement each function line ran
    for (sqlglot parses SQL templates and builds tables while its classes are created)."""
    if code.co_flags & CO_OPTIMIZED:
        frame = sys._getframe(2).f_back
        while frame is not None:
            fcode = frame.f_code
            if not fcode.co_flags & CO_OPTIMIZED and fcode.co_filename.startswith(ROOT):
                key = (_fid(path), line)
                seen = _origins.get(key)
                if seen is None:
                    seen = _origins[key] = set()
                seen.add((_fid(fcode.co_filename), frame.f_lineno))
                return
            frame = frame.f_back
    else:
        # a new import-time statement: let function lines report again on its behalf
        mon.restart_events()


def _audit(event, args):
    if event == "open":
        suffix = ""
    elif event in ("os.listdir", "os.scandir"):
        suffix = "/"
    else:
        return
    if not args:
        return
    path = args[0]
    if path is None:
        path = "."
    if isinstance(path, int):
        return
    try:
        path = os.path.abspath(os.fsdecode(path))
    except (TypeError, ValueError):
        return
    if path.startswith(ROOT):
        seen = _reads.get(_ctx)
        if seen is None:
            seen = _reads[_ctx] = set()
        seen.add(path[len(ROOT):] + suffix)


def _enter(ctx):
    global _ctx
    _ctx = ctx
    mon.restart_events()


def _flush(ctx):
    hits = _lines.pop(ctx, None)
    if hits:
        _out.write("C %s\t%s\n" % (ctx, ",".join("%d:%d" % h for h in sorted(hits))))
    reads = _reads.pop(ctx, None)
    if reads:
        _out.write("O %s\t%s\n" % (ctx, "|".join(sorted(reads))))
    missing = _missing.pop(ctx, None)
    if missing:
        names = sorted({"%s.%s:%s" % (cls.__module__, cls.__qualname__, arg) for cls, arg in missing})
        _out.write("V %s\t%s\n" % (ctx, ",".join(names)))
    attrs = _attr_reads.pop(ctx, None)
    if attrs:
        _out.write("R %s\t%s\n" % (ctx, ",".join(sorted(attrs))))
    keyed = _key_reads.pop(ctx, None)
    if keyed:
        import json

        _out.write("Y %s\t%s\n" % (ctx, json.dumps(sorted({(tag, _key_text(k)) for tag, k in keyed}))))
    dyn = _dyn_imports.pop(ctx, None)
    if dyn:
        _out.write("M %s\t%s\n" % (ctx, ",".join(sorted(dyn))))
    data = _data_reads.pop(ctx, None)
    if data:
        _out.write("D %s\t%s\n" % (ctx, ",".join(sorted(data))))
    users = _users.pop(ctx, None)
    if users:
        _out.write("U %s\t%s\n" % (ctx, ",".join(sorted({_key_of(c) for c in users if c in _class_keys}))))
    weak = _weak.pop(ctx, None)
    if weak:
        _out.write("W %s\t%s\n" % (ctx, ",".join(map(str, sorted(weak)))))


class InlineExecutor:
    """Stands in for ProcessPoolExecutor: runs each task when it is submitted, in this process."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **kwargs):
        import pickle

        if initializer is not None:
            initializer(*pickle.loads(pickle.dumps(initargs)))

    def submit(self, fn, /, *args, **kwargs):
        import concurrent.futures
        import pickle

        future = concurrent.futures.Future()
        try:
            fn, args, kwargs = pickle.loads(pickle.dumps((fn, args, kwargs)))
            future.set_result(pickle.loads(pickle.dumps(fn(*args, **kwargs))))
        except BaseException as error:  # noqa: BLE001 - the future carries it, like a real pool
            future.set_exception(error)
        return future

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        return [fn(*args) for args in zip(*iterables)]

    def shutdown(self, wait=True, *, cancel_futures=False):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _inline_process_pools():
    import concurrent.futures
    import concurrent.futures.process

    concurrent.futures.ProcessPoolExecutor = InlineExecutor
    concurrent.futures.process.ProcessPoolExecutor = InlineExecutor


def _import_everything():
    import importlib
    import pkgutil

    import sqlglot

    for info in pkgutil.walk_packages(sqlglot.__path__, "sqlglot."):
        if info.name.endswith(".__main__"):
            continue
        try:
            importlib.import_module(info.name)
        except Exception as error:  # noqa: BLE001
            print("trace_plugin: could not import %s: %r" % (info.name, error), file=sys.stderr)


def _record_missing_args():
    """Wraps Expression.error_messages to note which args were absent at validation time.

    ``arg_types`` values only matter through ``required_args``, and ``required_args`` is only
    read by ``error_messages``. Flipping ``arg_types[k]`` on a class therefore changes behaviour
    only where an instance of that class was validated while ``k`` was missing.
    """
    from sqlglot.expressions import core

    original = core.Expression.error_messages

    def error_messages(self, args=None):
        cls = type(self)
        values = self.args
        noted = None
        for key in cls.arg_types:
            value = values.get(key)
            if value is None or (isinstance(value, list) and not value):
                if noted is None:
                    noted = _missing.get(_ctx)
                    if noted is None:
                        noted = _missing[_ctx] = set()
                noted.add((cls, key))
        return original(self, args)

    core.Expression.error_messages = error_messages


# ----------------------------------------------------------------------------- attribute reads
# With PROFILE_READS=1 every data attribute of every sqlglot class (and every function stored
# on a class but defined elsewhere) is replaced by a descriptor that returns the same object and
# notes "<owner class>:<name>" for the current context. The owner is the class whose __dict__ holds the value, so an inherited read is
# charged to the class that defines it. Hot Expression attributes are left alone: arg_types and
# required_args are covered by the validation records and key is derived from the class name.

READS = os.environ.get("PROFILE_READS") == "1"
# With PROFILE_USERS=1 every class attribute is recorded (arg_types too), and so is every class
# a test instantiates or generates SQL for: "U <ctx>\t<class>,...". Edits nothing models more
# precisely are charged to the users of the class or module they sit in.
USERS = os.environ.get("PROFILE_USERS") == "1"
# With PROFILE_DATA=1: before every test the dialect registry goes back to what `import sqlglot`
# alone leaves in it, so a test that looks a dialect up by name loads it itself, and every
# importlib.import_module a test makes is recorded ("M <ctx>\t<module>,..."): a module that no
# longer imports fails exactly the tests that load it, even ones the registry used to spare. Lists
# a test class's setUpClass loads from a SQL fixture are served as recording copies
# ("P <tag>\t<fixture>\t<loader>", then "D <ctx>\t<tag>,..." for the tests that read them).
DATA = os.environ.get("PROFILE_DATA") == "1"
_dyn_imports = {}
_data_reads = {}
_registry_base = {}
_SKIP_EXPRESSION_ATTRS = set() if USERS else {"arg_types", "required_args", "key"}
_users = {}


def _note_user(cls):
    seen = _users.get(_ctx)
    if seen is None:
        seen = _users[_ctx] = set()
    seen.add(cls)


def _note(store, item):
    seen = store.get(_ctx)
    if seen is None:
        seen = store[_ctx] = set()
    seen.add(item)


def _record_dynamic_imports():
    import importlib

    from sqlglot.dialects.dialect import Dialect

    _registry_base.update({k: v for k, v in Dialect._classes.items() if v.__module__ == "sqlglot.dialects.dialect"})
    original = importlib.import_module

    def import_module(name, package=None):
        if package is None and (name == "sqlglot" or name.startswith("sqlglot.")):
            _note(_dyn_imports, name)
        return original(name, package)

    importlib.import_module = import_module


def _reset_dialect_registry():
    if not DATA:
        return
    dialect = sys.modules.get("sqlglot.dialects.dialect")
    if dialect is not None:
        classes = dialect.Dialect._classes
        classes.clear()
        classes.update(_registry_base)


class _DataList(list):
    """A list a test class's setUpClass loaded from a fixture: notes every test that reads it."""

    __slots__ = ("_tag",)

    def _seen(self):
        _note(_data_reads, self._tag)

    def __getitem__(self, i):
        self._seen()
        return list.__getitem__(self, i)

    def __iter__(self):
        self._seen()
        return list.__iter__(self)

    def __len__(self):
        self._seen()
        return list.__len__(self)

    def __contains__(self, x):
        self._seen()
        return list.__contains__(self, x)

    def __reversed__(self):
        self._seen()
        return list.__reversed__(self)

    def __eq__(self, other):
        self._seen()
        return list.__eq__(self, other)

    def __add__(self, other):
        self._seen()
        return list.__add__(self, other)

    def __repr__(self):
        self._seen()
        return list.__repr__(self)

    def __reduce_ex__(self, protocol):
        self._seen()
        return (list, (list(list.__iter__(self)),))

    def index(self, *args):
        self._seen()
        return list.index(self, *args)

    def count(self, x):
        self._seen()
        return list.count(self, x)

    def copy(self):
        self._seen()
        return list.copy(self)


def _serve_class_data(cls, before, fixture_ctx):
    """After a class-level fixture: lists it put on the test class that equal what a tests.helpers
    loader gives for a SQL fixture it read are replaced by recording copies."""
    try:
        from tests import helpers
    except Exception:  # noqa: BLE001
        return
    read = sorted(p for p in _reads.get(fixture_ctx, ()) if p.startswith("tests/fixtures/") and p.endswith(".sql"))
    if not read:
        return
    loaded = {}
    for path in read:
        rel = path[len("tests/fixtures/"):]
        for loader in ("load_sql_fixture_pairs", "load_sql_fixtures"):
            try:
                loaded[(path, loader)] = list(getattr(helpers, loader)(rel))
            except Exception:  # noqa: BLE001
                pass
    for name, value in list(vars(cls).items()):
        if type(value) is not list or before.get(name) == id(value):
            continue
        for (path, loader), items in loaded.items():
            if items == value:
                tag = "%s.%s:%s" % (cls.__module__, cls.__qualname__, name)
                data = _DataList(value)
                data._tag = tag
                setattr(cls, name, data)
                _out.write("P %s\t%s\t%s\n" % (tag, path, loader))
                break


def _record_users():
    from sqlglot import generator, parser, tokens
    from sqlglot.dialects import dialect
    from sqlglot.expressions import core

    def wrap_init(klass):
        original = klass.__init__

        def __init__(self, *args, **kwargs):
            _note_user(type(self))
            return original(self, *args, **kwargs)

        klass.__init__ = __init__

    for klass in (core.Expr, dialect.Dialect, tokens.Tokenizer, parser.Parser, generator.Generator):
        wrap_init(klass)
    original_sql = generator.Generator.sql

    def sql(self, expression, *args, **kwargs):
        if isinstance(expression, core.Expr):
            _note_user(type(expression))
        return original_sql(self, expression, *args, **kwargs)

    generator.Generator.sql = sql
_attr_reads = {}
_installed = []
_class_keys = {}


class _Recorded:
    __slots__ = ("tag", "value")

    def __init__(self, tag, value):
        self.tag = tag
        self.value = value

    def __get__(self, obj, objtype=None):
        seen = _attr_reads.get(_ctx)
        if seen is None:
            seen = _attr_reads[_ctx] = set()
        seen.add(self.tag)
        return self.value


class _RecordedBinding(_Recorded):
    """For functions stored on a class but made elsewhere (factory closures, helpers, static
    and class methods wrapping them): records the read, then binds like the plain attribute."""

    __slots__ = ()

    def __get__(self, obj, objtype=None):
        seen = _attr_reads.get(_ctx)
        if seen is None:
            seen = _attr_reads[_ctx] = set()
        seen.add(self.tag)
        return self.value.__get__(obj, objtype)


# With PROFILE_KEYS=1 (and PROFILE_READS=1), a class attribute holding a plain dict is served
# as a copy that also notes which keys each context looks up. Any use that may touch every key
# (iteration, items(), copies, ** spreads, comparisons, repr) is noted as "*".
KEYS = os.environ.get("PROFILE_KEYS") == "1"
_key_reads = {}


_suppressed = [False]


def _note_key(tag, key):
    if _suppressed[0]:
        return
    seen = _key_reads.get(_ctx)
    if seen is None:
        seen = _key_reads[_ctx] = set()
    try:
        seen.add((tag, key))
    except TypeError:  # unhashable key: the lookup itself would have failed
        seen.add((tag, _WHOLE))


_WHOLE = object()


class _KeyRecordingDict(dict):
    __slots__ = ("_tag",)

    def __getitem__(self, key):
        _note_key(self._tag, key)
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        _note_key(self._tag, key)
        return dict.get(self, key, default)

    def __contains__(self, key):
        _note_key(self._tag, key)
        return dict.__contains__(self, key)

    def pop(self, key, *default):
        _note_key(self._tag, key)
        return dict.pop(self, key, *default)

    def setdefault(self, key, default=None):
        _note_key(self._tag, key)
        return dict.setdefault(self, key, default)

    def _all(self):
        _note_key(self._tag, _WHOLE)

    def __iter__(self):
        self._all()
        return dict.__iter__(self)

    def __reversed__(self):
        self._all()
        return dict.__reversed__(self)

    def keys(self):
        self._all()
        return dict.keys(self)

    def values(self):
        self._all()
        return dict.values(self)

    def items(self):
        self._all()
        return dict.items(self)

    def __len__(self):
        self._all()
        return dict.__len__(self)

    def copy(self):
        self._all()
        return dict.copy(self)

    def __eq__(self, other):
        self._all()
        return dict.__eq__(self, other)

    def __ne__(self, other):
        self._all()
        return dict.__ne__(self, other)

    def __or__(self, other):
        self._all()
        return dict.__or__(self, other)

    def __ror__(self, other):
        self._all()
        return dict.__ror__(self, other)

    def __repr__(self):
        self._all()
        return dict.__repr__(self)

    def __reduce_ex__(self, protocol):
        self._all()
        return (dict, (dict(dict.items(self)),))

    __hash__ = None


def _key_text(key):
    """A stable spelling of a dict key, shared with the map builder."""
    import enum

    if key is _WHOLE:
        return "*"
    if isinstance(key, bool) or key is None:
        return "k:%r" % (key,)
    if isinstance(key, (str, int, float)):
        return "%s:%r" % (type(key).__name__, key)
    if isinstance(key, enum.Enum):
        return "e:%s.%s.%s" % (type(key).__module__, type(key).__qualname__, key.name)
    if isinstance(key, type):
        return "c:%s.%s" % (key.__module__, key.__qualname__)
    if isinstance(key, tuple):
        return "t:(%s)" % ",".join(_key_text(k) for k in key)
    return "*"


def _recorder_for(cls, name, value, tag):
    import types

    kind = type(value)
    if isinstance(value, type):
        return None
    if KEYS and kind is dict:
        served = _KeyRecordingDict(value)
        served._tag = tag
        return _Recorded(tag, served)
    if not hasattr(kind, "__get__"):
        return _Recorded(tag, value)
    if hasattr(kind, "__set__") or hasattr(kind, "__delete__"):
        return None  # properties and slots: data descriptors keep their own semantics
    func = value.__func__ if isinstance(value, (staticmethod, classmethod)) else value
    if isinstance(func, types.FunctionType) and func.__qualname__.startswith(cls.__qualname__ + "."):
        return None  # a method written in this class body; edits to it are runtime edits
    return _RecordedBinding(tag, value)


def _sqlglot_classes():
    import enum

    found = {}
    stack = []
    for name, module in list(sys.modules.items()):
        if name == "sqlglot" or name.startswith("sqlglot."):
            for value in list(vars(module).values()):
                if isinstance(value, type) and (value.__module__ or "").startswith("sqlglot"):
                    stack.append(value)
    while stack:
        cls = stack.pop()
        if id(cls) in found or issubclass(cls, (enum.Enum, BaseException, type)):
            continue
        if not (cls.__module__ or "").startswith("sqlglot"):
            continue
        found[id(cls)] = cls
        stack.extend(v for v in vars(cls).values() if isinstance(v, type))
        stack.extend(type.__subclasses__(cls))
    return list(found.values())


def _key_of(cls):
    key = _class_keys.get(cls)
    if key is None:
        key = "%s.%s" % (cls.__module__, cls.__qualname__)
        if key in _class_keys.values():
            key = "%s#%d" % (key, len(_class_keys))
        _class_keys[cls] = key
    return key


def _install_read_recorders():
    from sqlglot.expressions import core

    classes = sorted(_sqlglot_classes(), key=lambda c: (c.__module__, c.__qualname__))
    for cls in classes:
        _key_of(cls)
    for cls in classes:
        skip = _SKIP_EXPRESSION_ATTRS if issubclass(cls, core.Expr) else ()
        owner = _key_of(cls)
        for name, value in list(vars(cls).items()):
            if name.startswith("__") and name.endswith("__"):
                continue
            if name in skip:
                continue
            desc = _recorder_for(cls, name, value, "%s:%s" % (owner, name))
            if desc is None:
                continue
            try:
                type.__setattr__(cls, name, desc)
            except (TypeError, AttributeError):
                continue
            _installed.append((cls, name, desc))
    by_owner = {}
    for cls, name, _desc in _installed:
        by_owner.setdefault(_key_of(cls), []).append(name)
    for owner, names in sorted(by_owner.items()):
        _out.write("A %s\t%s\n" % (owner, ",".join(sorted(names))))
    # The family relations the map builder needs: MROs and each dialect's component classes.
    for cls in classes:
        mro = "|".join(_key_of(k) for k in cls.__mro__ if k in _class_keys)
        parts = []
        for attr in ("tokenizer_class", "jsonpath_tokenizer_class", "parser_class", "generator_class"):
            comp = cls.__dict__.get(attr)
            if isinstance(comp, _Recorded):
                comp = comp.value
            if isinstance(comp, type) and comp in _class_keys:
                parts.append(_key_of(comp))
        _out.write("K %s\t%s\t%s\n" % (_key_of(cls), mro, "|".join(parts)))


def _reset_runtime_caches():
    """Caches shared across tests would charge their work to whichever test filled them first.
    sqlglot.generator caches each generator class's dispatch table (TRANSFORMS plus the *_sql
    methods) the first time one is built; clearing such caches before every test makes each
    test do (and be charged for) its own."""
    gen = sys.modules.get("sqlglot.generator")
    cache = getattr(gen, "_DISPATCH_CACHE", None)
    if isinstance(cache, dict):
        cache.clear()
    # TestExecutor keeps duckdb's answers per SQL text in a class-level dict shared by the TPC-H
    # tests; without clearing it only the first of them would run transpile() for each query.
    executor_tests = sys.modules.get("tests.test_executor")
    shared = getattr(getattr(executor_tests, "TestExecutor", None), "cache", None)
    if isinstance(shared, dict):
        shared.clear()


def _record_dispatch_keys():
    """Serve each dispatch table as a key-recording copy tagged like the TRANSFORMS it came from,
    so a test is charged for the expression types it generated, not for the bulk copy of
    TRANSFORMS made while building the table."""
    gen = sys.modules.get("sqlglot.generator")
    if gen is None or not hasattr(gen, "_build_dispatch"):
        return
    original = gen._build_dispatch

    def build(cls):
        _suppressed[0] = True
        try:
            dispatch = original(cls)
        finally:
            _suppressed[0] = False
        for klass in cls.__mro__:
            desc = klass.__dict__.get("TRANSFORMS")
            if isinstance(desc, _Recorded):
                served = _KeyRecordingDict(dispatch)
                served._tag = desc.tag
                return served
        return dispatch

    gen._build_dispatch = build


def _reinstall_read_recorders():
    """Tests that patch a class attribute restore the plain value; put the recorder back."""
    for cls, name, desc in _installed:
        current = cls.__dict__.get(name, desc)
        if current is not desc and current is desc.value:
            type.__setattr__(cls, name, desc)


# ----------------------------------------------------------------------------- weak mutation
# With PROFILE_WEAK=1 sqlglot is imported through a loader that rewrites comparisons and boolean
# operators (see weak_rewrite.py), and line tracing is off. Records:
#     W <ctx>\t<site*4+variant>,...   swaps that would have changed a result in that context
#     S\t<json site table>

WEAK = os.environ.get("PROFILE_WEAK") == "1"
_weak = {}


def _weak_set():
    seen = _weak.get(_ctx)
    if seen is None:
        seen = _weak[_ctx] = set()
    return seen


def _install_weak_loader():
    import importlib.machinery
    import importlib.util

    import weak_rewrite

    prefix = ROOT + "sqlglot" + os.sep

    class WeakLoader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            path = self.get_filename(fullname)
            if not path.startswith(prefix):
                return super().get_code(fullname)
            source = importlib.util.decode_source(self.get_data(path))
            return weak_rewrite.rewrite(source, path[len(ROOT):], path)

    m = importlib.machinery
    hook = m.FileFinder.path_hook(
        (m.ExtensionFileLoader, m.EXTENSION_SUFFIXES),
        (WeakLoader, m.SOURCE_SUFFIXES),
        (m.SourcelessFileLoader, m.BYTECODE_SUFFIXES),
    )
    sys.path_hooks.insert(0, hook)
    sys.path_importer_cache.clear()
    weak_rewrite.install(_weak_set)


def pytest_configure(config):
    global _out
    _inline_process_pools()
    _out = gzip.open(OUT, "wt", compresslevel=6)
    mon.use_tool_id(TOOL, "trace_plugin")
    mon.register_callback(TOOL, mon.events.LINE, _on_line)
    if WEAK:
        _install_weak_loader()
    else:
        mon.set_events(TOOL, mon.events.LINE)
    sys.addaudithook(_audit)
    _enter("<import>")
    _import_everything()
    _record_missing_args()
    if READS:
        _install_read_recorders()
        if KEYS:
            _record_dispatch_keys()
    if USERS:
        _record_users()
    if DATA:
        _record_dynamic_imports()
    _flush("<import>")
    _enter("<other>")


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    if not isinstance(collector, pytest.Module):
        yield
        return
    prev = _ctx
    _enter("<collect>|" + collector.nodeid)
    yield
    _enter(prev)


@pytest.hookimpl(hookwrapper=True)
def pytest_fixture_setup(fixturedef, request):
    global _ctx
    if fixturedef.scope == "function":
        yield
        return
    prev = _ctx
    ctx = "<fixture>|%s|%s" % (fixturedef.scope, request.node.nodeid)
    cls = getattr(request, "cls", None) if fixturedef.scope == "class" else None
    before = {k: id(v) for k, v in vars(cls).items()} if DATA and cls is not None else None
    _enter(ctx)
    yield
    if before is not None:
        _serve_class_data(cls, before, ctx)
    # Lines that ran inside the fixture stay disabled for the rest of this phase; they are
    # charged to every test in the fixture's scope when the map is built.
    _ctx = prev


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    _reset_runtime_caches()
    _reset_dialect_registry()
    _enter(item.nodeid + "|setup")
    yield
    _flush(item.nodeid + "|setup")


# test_execute_tpcds runs ~100 TPC-DS queries, each parsed, sent through a process pool,
# executed and compared with duckdb's answer for sqlglot's own transpiled SQL. Its call phase is
# split into one context per query ("<nodeid>|call#q<k>"), so the selector can re-check just
# the queries an edit touches instead of paying for the whole test.
PER_QUERY = {"tests/test_executor.py::TestExecutor::test_execute_tpcds"}


class _PerQuery:
    def __init__(self, item):
        self.item = item
        self.base = item.nodeid + "|call"
        self.count = 0
        self.patches = []

    def __enter__(self):
        import unittest

        module = self.item.module
        original_parse = module.parse_one
        original_subtest = unittest.TestCase.subTest
        owner = self

        def parse_one(sql, *args, **kwargs):
            # the test parses query k right before submitting it to the pool
            _enter("%s#q%d" % (owner.base, owner.count))
            owner.count += 1
            return original_parse(sql, *args, **kwargs)

        def sub_test(case, msg=None, **params):
            label = str(msg or "")
            if label.startswith("tpc-ds ") and label[7:].isdigit():
                return owner.scoped("%s#q%d" % (owner.base, int(label[7:]) - 1), original_subtest(case, msg, **params))
            return original_subtest(case, msg, **params)

        module.parse_one = parse_one
        unittest.TestCase.subTest = sub_test
        self.patches = [(module, "parse_one", original_parse), (unittest.TestCase, "subTest", original_subtest)]
        return self

    def scoped(self, ctx, inner):
        import contextlib

        @contextlib.contextmanager
        def manager():
            _enter(ctx)
            try:
                with inner:
                    yield
            finally:
                _enter(self.base)  # anything between subtests belongs to the whole test

        return manager()

    def __exit__(self, *exc):
        for obj, name, value in self.patches:
            setattr(obj, name, value)
        return False


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    _enter(item.nodeid + "|call")
    if item.nodeid in PER_QUERY:
        with _PerQuery(item):
            yield
        _enter(item.nodeid + "|call")
    else:
        yield
    for ctx in [c for c in set(_lines) | set(_weak) | set(_attr_reads) | set(_reads) | set(_key_reads) | set(_users) | set(_dyn_imports) | set(_data_reads) if c.startswith(item.nodeid + "|call")]:
        _flush(ctx)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    _enter(item.nodeid + "|teardown")
    yield
    _flush(item.nodeid + "|teardown")
    if READS:
        _reinstall_read_recorders()
    _enter("<other>")


def pytest_sessionfinish(session, exitstatus):
    mon.set_events(TOOL, 0)
    for ctx in list(set(_lines) | set(_reads) | set(_missing) | set(_attr_reads) | set(_weak) | set(_key_reads) | set(_users) | set(_dyn_imports) | set(_data_reads)):
        _flush(ctx)
    if WEAK:
        import json

        import weak_rewrite

        _out.write("S\t%s\n" % json.dumps(weak_rewrite.SITES))
    for (fid, line), origins in _origins.items():
        _out.write("I %d:%d\t%s\n" % (fid, line, ",".join("%d:%d" % o for o in sorted(origins))))
    for path, fid in _files.items():
        rel = path[len(ROOT):] if path.startswith(ROOT) else path
        _out.write("F %d\t%s\n" % (fid, rel))
    _out.close()
