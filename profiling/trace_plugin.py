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
_SKIP_EXPRESSION_ATTRS = {"arg_types", "required_args", "key"}
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


def _recorder_for(cls, name, value, tag):
    import types

    kind = type(value)
    if isinstance(value, type):
        return None
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


def _reinstall_read_recorders():
    """Tests that patch a class attribute restore the plain value; put the recorder back."""
    for cls, name, desc in _installed:
        current = cls.__dict__.get(name, desc)
        if current is not desc and current is desc.value:
            type.__setattr__(cls, name, desc)


def pytest_configure(config):
    global _out
    _inline_process_pools()
    _out = gzip.open(OUT, "wt", compresslevel=6)
    mon.use_tool_id(TOOL, "trace_plugin")
    mon.register_callback(TOOL, mon.events.LINE, _on_line)
    mon.set_events(TOOL, mon.events.LINE)
    sys.addaudithook(_audit)
    _enter("<import>")
    _import_everything()
    _record_missing_args()
    if READS:
        _install_read_recorders()
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
    _enter("<fixture>|%s|%s" % (fixturedef.scope, request.node.nodeid))
    yield
    # Lines that ran inside the fixture stay disabled for the rest of this phase; they are
    # charged to every test in the fixture's scope when the map is built.
    _ctx = prev


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    _enter(item.nodeid + "|setup")
    yield
    _flush(item.nodeid + "|setup")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    _enter(item.nodeid + "|call")
    yield
    _flush(item.nodeid + "|call")


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
    for ctx in list(set(_lines) | set(_reads) | set(_missing) | set(_attr_reads)):
        _flush(ctx)
    for (fid, line), origins in _origins.items():
        _out.write("I %d:%d\t%s\n" % (fid, line, ",".join("%d:%d" % o for o in sorted(origins))))
    for path, fid in _files.items():
        rel = path[len(ROOT):] if path.startswith(ROOT) else path
        _out.write("F %d\t%s\n" % (fid, rel))
    _out.close()
