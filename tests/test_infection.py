import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from infection import unchanged


class TestInfection(unittest.TestCase):
    def test_filter_checks_every_observation(self):
        old = "if x == 2:"
        site = [(5, 7, 8, 9, False, {(1,): 1, (2, 8): 2, (3,): 4})]
        self.assertEqual(unchanged(site, old, "if x == 8:"), 5)
        self.assertEqual(unchanged(site, old, "if x >= 2:"), 1)
        self.assertEqual(unchanged(site, old, "if y == 8:"), 0)
        self.assertEqual(unchanged(site, old, "if x < 8:"), 0)

    def test_literal_on_left(self):
        site = [(5, 6, 3, 4, True, {(2,): 1, (4,): 2})]
        self.assertEqual(unchanged(site, "if 2 < x:", "if 2 <= x:"), 2)

    def test_relations_and_custom_operands(self):
        import build_infection as build
        from infection import OPS
        key = ("site",)
        build.observed.clear()
        build.compare(key, "<", 1, 2)
        build.compare(key, "<", 2, 2)
        site = [(5, 6, None, None, False, {tuple(build.observed[key]): 1})]
        self.assertEqual(unchanged(site, "if x < y:", "if x <= y:"), 0)
        self.assertEqual(unchanged(site, "if x < y:", "if x != y:"), 1)
        self.assertIn("is", OPS)
        build.compare(key, "==", object(), object())
        self.assertIsNone(build.observed[key])
