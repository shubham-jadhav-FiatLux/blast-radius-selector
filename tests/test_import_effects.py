"""Module globals are owned by their module: two modules' tables of the same name do not share
readers, and a table reaches the class body that copies it only through the module that
imports it."""

import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "profiling"))

from import_effects import GLOBAL, ImportEffects, absolute, mkey  # noqa: E402


class TestAbsolute(unittest.TestCase):
    def test_relative_imports(self):
        self.assertEqual(absolute("sqlglot.typing.mysql", False, 0, "sqlglot.typing"), "sqlglot.typing")
        self.assertEqual(absolute("sqlglot.typing.mysql", False, 1, ""), "sqlglot.typing")
        self.assertEqual(absolute("sqlglot.typing.mysql", False, 1, "spark"), "sqlglot.typing.spark")
        self.assertEqual(absolute("sqlglot.typing", True, 1, "spark"), "sqlglot.typing.spark")
        self.assertEqual(absolute("sqlglot.typing.mysql", False, 2, "dialects"), "sqlglot.dialects")
        self.assertIsNone(absolute("top", False, 1, "x"))


def _index(derivations=(), binds=()):
    return {"derivations": list(derivations), "import_binds": list(binds)}


class TestModuleOwners(unittest.TestCase):
    def setUp(self):
        # sqlglot.typing.a and sqlglot.typing.b each define TABLE; sqlglot.dialects.a imports
        # the first and copies it into class A's body.
        self.added = []
        for name in ("sqlglot.typing.a", "sqlglot.typing.b", "sqlglot.dialects.a"):
            if name not in sys.modules:
                sys.modules[name] = types.ModuleType(name)
                self.added.append(name)
        files = {
            "sqlglot/typing/a.py": _index(),
            "sqlglot/typing/b.py": _index(),
            "sqlglot/dialects/a.py": _index(
                derivations=[("class", "A", ("TABLE",), ("TABLE",), ())],
                binds=[(0, "sqlglot.typing.a", "TABLE", "TABLE")],
            ),
        }
        families = {"sqlglot.dialects.a.A": (("sqlglot.dialects.a.A",), ())}
        instrumented = {("sqlglot.dialects.a.A", "TABLE")}
        reads = {"sqlglot.dialects.a.A:TABLE": 0b100}
        readers = {("sqlglot.typing.a", "TABLE"): 0b001, ("sqlglot.typing.b", "TABLE"): 0b010}
        self.effects = ImportEffects(files, families, instrumented, reads, set(),
                                     lambda module, name: readers.get((module, name), 0))

    def tearDown(self):
        for name in self.added:
            del sys.modules[name]

    def test_copy_reaches_the_receiving_class_only(self):
        pairs = self.effects.affected(mkey("sqlglot.typing.a"), "TABLE")
        self.assertIn(("sqlglot.dialects.a.A", "TABLE"), pairs)
        self.assertIn((mkey("sqlglot.dialects.a"), "TABLE"), pairs)
        self.assertNotIn((mkey("sqlglot.typing.b"), "TABLE"), pairs)

    def test_readers_are_per_module(self):
        self.assertEqual(self.effects.tests_for(mkey("sqlglot.typing.a"), "TABLE"), 0b101)
        self.assertEqual(self.effects.tests_for(mkey("sqlglot.typing.b"), "TABLE"), 0b010)

    def test_unprofiled_class_attribute_runs_everything(self):
        self.assertEqual(self.effects.tests_for("sqlglot.dialects.a.B", "TABLE"), GLOBAL)


if __name__ == "__main__":
    unittest.main()
