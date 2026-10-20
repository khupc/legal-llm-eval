"""Check the statistics against published reference values.
Run: python -m unittest discover -s tests
"""
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from agreement import cohen_kappa, fleiss_kappa  # noqa: E402
from common import macro_f1, per_class  # noqa: E402
from compare import holm, mcnemar_exact  # noqa: E402
from parse_outputs import is_degenerate  # noqa: E402


class TestAgreement(unittest.TestCase):
    def test_fleiss_wikipedia_example(self):
        # Fleiss (1971) worked example as reproduced on Wikipedia: kappa = 0.210
        rows = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0],
                [2, 2, 8, 1, 1], [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2],
                [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
        labels = list(range(5))
        table = {i: Counter({c: n for c, n in zip(labels, r)}) for i, r in enumerate(rows)}
        self.assertAlmostEqual(fleiss_kappa(table, labels), 0.210, places=3)

    def test_cohen_perfect_and_chance(self):
        a = {i: "x" if i % 2 else "y" for i in range(10)}
        self.assertAlmostEqual(cohen_kappa(a, a), 1.0)
        # 2x2 table [[20, 5], [10, 15]]: po = 0.70, pe = 0.50 -> kappa = 0.40
        pairs = [("y", "y")] * 20 + [("y", "n")] * 5 + [("n", "y")] * 10 + [("n", "n")] * 15
        r1 = {i: p[0] for i, p in enumerate(pairs)}
        r2 = {i: p[1] for i, p in enumerate(pairs)}
        self.assertAlmostEqual(cohen_kappa(r1, r2), 0.40)


class TestComparisons(unittest.TestCase):
    def test_mcnemar_exact(self):
        # b=1, c=9: two-sided exact p = 2 * P(X<=1 | n=10, 0.5) = 22/1024
        self.assertAlmostEqual(mcnemar_exact(1, 9), 22 / 1024)
        self.assertEqual(mcnemar_exact(0, 0), 1.0)

    def test_holm(self):
        adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(adj["a"], 0.03)
        self.assertAlmostEqual(adj["c"], 0.06)
        self.assertAlmostEqual(adj["b"], 0.06)  # monotone step-down


class TestMetrics(unittest.TestCase):
    def test_unparseable_counts_against_recall_only(self):
        gold = {1: "a", 2: "a", 3: "b", 4: "b"}
        pred = {1: "a", 2: "", 3: "b", 4: "a"}
        s = per_class(gold, pred, ["a", "b"], gold)
        self.assertAlmostEqual(s["a"]["precision"], 0.5)
        self.assertAlmostEqual(s["a"]["recall"], 0.5)
        self.assertAlmostEqual(s["b"]["precision"], 1.0)
        self.assertAlmostEqual(s["b"]["recall"], 0.5)
        self.assertAlmostEqual(macro_f1(s), (0.5 + 2 / 3) / 2)



class TestParser(unittest.TestCase):
    def test_degenerate_loop_detected(self):
        self.assertTrue(is_degenerate("字" * 80))
        self.assertFalse(is_degenerate("Reasoning... Answer: loan"))


if __name__ == "__main__":
    unittest.main()
