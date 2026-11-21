"""Unit tests for the selector's building blocks. Run with: python -m unittest discover tests"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import blast_select as bs  # noqa: E402

PATCH = """diff --git a/sqlglot/x.py b/sqlglot/x.py
index 1111111..2222222 100644
--- a/sqlglot/x.py
+++ b/sqlglot/x.py
@@ -10,7 +10,7 @@ class A:
     def f(self):
         a = 1
         b = 2
-        if a < b:
+        if a <= b:
             return a
         return b

diff --git a/tests/fixtures/q.sql b/tests/fixtures/q.sql
--- a/tests/fixtures/q.sql
+++ b/tests/fixtures/q.sql
@@ -3,2 +3,3 @@
 SELECT 1;
+-- a comment that starts with dashes
 SELECT 2;
"""


class TestPatchParsing(unittest.TestCase):
    def test_blocks_and_line_numbers(self):
        changes = bs.parse_patch(PATCH)
        self.assertEqual([c.path for c in changes], ["sqlglot/x.py", "tests/fixtures/q.sql"])
        block = changes[0].blocks[0]
        self.assertEqual((block.start, block.old, block.new), (13, ["        if a < b:"], ["        if a <= b:"]))
        insert = changes[1].blocks[0]
        # a "--" line inside a hunk is content, not a file header
        self.assertEqual((insert.start, insert.old, insert.new), (4, [], ["-- a comment that starts with dashes"]))

    def test_new_and_deleted_files(self):
        text = "diff --git a/n.py b/n.py\nnew file mode 100644\n--- /dev/null\n+++ b/n.py\n@@ -0,0 +1 @@\n+x = 1\n"
        (change,) = bs.parse_patch(text)
        self.assertEqual((change.status, change.path, change.blocks[0].start), ("new", "n.py", 1))
        text = "diff --git a/d.py b/d.py\ndeleted file mode 100644\n--- a/d.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x = 1\n"
        (change,) = bs.parse_patch(text)
        self.assertEqual(change.status, "delete")


class TestComments(unittest.TestCase):
    def test_strip_comment_ignores_hashes_in_strings(self):
        self.assertEqual(bs.strip_comment("x = '#not a comment'  # real"), "x = '#not a comment'  ")
        self.assertEqual(bs.strip_comment('y = """#"""'), 'y = """#"""')
        self.assertEqual(bs.strip_comment("    # only a comment"), "    ")

    def test_code_lines_drop_comments_and_blanks(self):
        self.assertEqual(bs.code_lines(["", "    # Cheap check first", "x = 1  # set x"]), ["x = 1"])


class TestColumns(unittest.TestCase):
    def test_changed_columns_widen_to_words(self):
        a, b, old, new = bs.changed_columns('    "constraint": False,', '    "constraint": True,')
        self.assertEqual((old, new), ("False", "True"))
        self.assertEqual('    "constraint": False,'[a:b], "False")
        _, _, old, new = bs.changed_columns("    WEEK_OFFSET = -1", "    WEEK_OFFSET = -2")
        self.assertEqual((old, new), ("1", "2"))


class TestWeakMatch(unittest.TestCase):
    def test_operator_and_constant_swaps(self):
        line = "        if self.x is Strategy.UPPER and len(args) > 1:"
        c0 = line.index(" is ") + 1
        k0 = line.index("1:")
        sites = [(c0, c0 + 2, "is", [("==", 7)]), (k0, k0 + 1, "1", [("2", 8), ("0", 9)])]
        self.assertEqual(bs.Selector.weak_match(sites, line, line.replace(" is ", " == ")), 7)
        self.assertEqual(bs.Selector.weak_match(sites, line, line.replace("> 1", "> 2")), 8)
        self.assertIsNone(bs.Selector.weak_match(sites, line, line.replace(" and ", " or ")))

    def test_dropped_not(self):
        line = "    if not after_dot:"
        c0 = line.index("not ")
        sites = [(c0, c0 + 4, "not ", [("", 3)])]
        self.assertEqual(bs.Selector.weak_match(sites, line, "    if after_dot:"), 3)


class TestPacking(unittest.TestCase):
    def cost(self, workers, ms):
        return max(sum(ms[t] for t in w) for w in workers) + 2000 * len(workers)

    def test_empty(self):
        self.assertEqual(bs.split([], {}), [])

    def test_one_long_test_dominates(self):
        ms = {0: 28800, 1: 900, 2: 800, 3: 700}
        workers = bs.split([0, 1, 2, 3], ms)
        self.assertEqual(self.cost(workers, ms), 28800 + 2 * 2000)

    def test_balanced_split_beats_a_single_worker(self):
        ms = {i: 5000 for i in range(8)}
        workers = bs.split(list(ms), ms)
        self.assertEqual(sorted(t for w in workers for t in w), list(range(8)))
        self.assertLessEqual(self.cost(workers, ms), 4 * 5000 + 2 * 2000)


if __name__ == "__main__":
    unittest.main()
