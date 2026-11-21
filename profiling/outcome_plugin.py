"""pytest plugin that writes each test's outcome the way the track labels it.

A test fails if any of its reports fails: setup, call, teardown or one of its subtests,
including timeouts. A test module that cannot be imported fails every test in it.
Output: JSON at ``OUTCOME_OUT``.
"""

import json
import os
import time

_results = {}
_collect_errors = []
_started = time.time()


def pytest_runtest_logreport(report):
    entry = _results.setdefault(report.nodeid, {"failed": False, "skipped": False, "seconds": 0.0})
    if report.failed:
        entry["failed"] = True
    if report.skipped and report.when in ("setup", "call"):
        entry["skipped"] = True
    if type(report).__name__ == "TestReport":
        entry["seconds"] += report.duration or 0.0


def pytest_collectreport(report):
    if report.failed:
        _collect_errors.append(report.nodeid)


def pytest_sessionfinish(session, exitstatus):
    if hasattr(session.config, "workerinput"):
        return  # pytest-xdist worker: the controller receives every report and writes the file
    with open(os.environ.get("OUTCOME_OUT", "outcomes.json"), "w") as f:
        json.dump(
            {
                "results": _results,
                "collect_errors": _collect_errors,
                "wall_seconds": round(time.time() - _started, 1),
                "exitstatus": int(exitstatus),
            },
            f,
        )
