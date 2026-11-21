"""pytest plugin that records what every sqlglot test touches.

Loaded with ``-p trace_plugin`` while running the sqlglot suite at the pinned commit.
It writes one gzip text file (``PROFILE_OUT``) with these records:

    F <fid>\t<path>               file id table, paths relative to the source root
    C <ctx>\t<fid>:<line>,...     lines executed in a context (negative: module or class body)
    O <ctx>\t<path>|<path>|...    files opened, or directories listed (trailing "/")
    V <ctx>\t<class>:<arg>,...    expression args that were missing when the node was validated

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
    return mon.DISABLE


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
    _enter("<other>")


def pytest_sessionfinish(session, exitstatus):
    mon.set_events(TOOL, 0)
    for ctx in list(set(_lines) | set(_reads) | set(_missing)):
        _flush(ctx)
    for path, fid in _files.items():
        rel = path[len(ROOT):] if path.startswith(ROOT) else path
        _out.write("F %d\t%s\n" % (fid, rel))
    _out.close()
