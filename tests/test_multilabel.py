"""Tests for multi-label support, label groups, clusters and contrast options.
Run: python -m unittest discover -s tests
"""
import json
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import agreement  # noqa: E402
import compare  # noqa: E402
import evaluate  # noqa: E402
import parse_outputs  # noqa: E402
from common import UNPARSEABLE, parse_cell, per_label  # noqa: E402
from route import vote  # noqa: E402
from test_robustness import TmpDir, run  # noqa: E402

S = frozenset


class TestSets(unittest.TestCase):
    def test_parse_cell(self):
        self.assertEqual(parse_cell("a"), S({"a"}))
        self.assertEqual(parse_cell("b|a"), S({"a", "b"}))
        self.assertEqual(parse_cell("none"), S())
        self.assertIs(parse_cell(""), UNPARSEABLE)

    def test_per_label_multilabel(self):
        pairs = [(S({"a", "b"}), S({"a"})),   # b missed
                 (S(), S({"b"})),             # b false alarm
                 (S({"a"}), UNPARSEABLE)]     # a missed
        s = per_label(pairs, ["a", "b"])
        self.assertAlmostEqual(s["a"]["precision"], 1.0)
        self.assertAlmostEqual(s["a"]["recall"], 0.5)
        self.assertAlmostEqual(s["b"]["precision"], 0.0)
        self.assertAlmostEqual(s["b"]["recall"], 0.0)


class TestVote(unittest.TestCase):
    def test_multilabel_vote(self):
        maj, contested = vote([S({"a"}), S({"a", "b"}), S({"a"})], True)
        self.assertEqual(maj, S({"a"}))
        self.assertEqual(contested, {"b"})
        maj, _ = vote([S({"a"}), S({"b"})], True)       # each label has exactly half
        self.assertIs(maj, UNPARSEABLE)

    def test_single_label_vote(self):
        maj, contested = vote([S({"a"}), S({"a"}), S({"b"})], False)
        self.assertEqual(maj, S({"a"}))
        self.assertEqual(contested, {"a", "b"})


class TestEvaluateModes(TmpDir):
    def test_multilabel_report(self):
        gold = self.csv("g.csv", "item_id,label\n1,a|b\n2,none\n3,a\n")
        pred = self.csv("p.csv", "item_id,config,label\n1,c,a\n2,c,none\n3,c,a\n")
        out, _ = run(evaluate.main, [gold, pred])
        self.assertIn("mode: multi-label", out)
        self.assertIn("exact match 0.667", out)
        self.assertIn("predicted 0.67 vs gold 1.00", out)

    def test_group_level(self):
        gold = self.csv("g.csv", "item_id,label\n1,ok\n2,spam\n3,scam\n")
        pred = self.csv("p.csv", "item_id,config,label\n1,c,ok\n2,c,scam\n3,c,ok\n")
        groups = self.csv("gr.csv", "label,group\nok,ok\nspam,bad\nscam,bad\n")
        out, _ = run(evaluate.main, [gold, pred, "--groups", groups])
        block = out.split("[group level]")[1]
        bad = next(l for l in block.splitlines() if l.startswith("bad"))
        self.assertEqual(bad.split()[1:4], ["1.000", "0.500", "0.667"])  # spam->scam is right at group level


class TestCompareOptions(TmpDir):
    def test_threshold_required(self):
        gold = self.csv("g.csv", "item_id,label\n1,a\n2,b\n")
        pred = self.csv("p.csv", "item_id,config,label\n1,x,a\n2,x,b\n1,y,a\n2,y,a\n")
        with self.assertRaises(SystemExit):
            run(compare.main, [gold, pred, "--contrast", "classes"])
        out, _ = run(compare.main, [gold, pred])                     # McNemar only
        self.assertNotIn("bootstrap", out)
        out, _ = run(compare.main, [gold, pred, "--contrast", "configs", "--threshold", "0.1", "--boot", "50"])
        self.assertIn("x -> y", out)

    def test_cluster_resample_keeps_clusters_whole(self):
        groups = {"k1": ["1", "2", "3"], "k2": ["4"], "k3": ["5", "6"]}
        sample = compare.cluster_resample(groups, random.Random(0))
        for members in groups.values():
            for m in members:  # a cluster is drawn all or nothing, never split
                self.assertEqual(sample.count(m), sample.count(members[0]))


class TestOtherScripts(TmpDir):
    def test_agreement_multilabel(self):
        ann = self.csv("a.csv", "item_id,annotator,label\n"
                                "1,A,a|b\n1,B,a|b\n2,A,none\n2,B,a\n3,A,b\n3,B,b\n4,A,a\n4,B,\n")
        out, _ = run(agreement.main, ann)
        self.assertIn("(multi-label)", out)
        self.assertIn("abstentions (empty label, treated as not annotated): 1", out)
        self.assertIn("raw agreement (identical label sets): 0.667", out)
        self.assertIn("Fleiss' kappa per label", out)

    def test_parse_multilabel(self):
        cfg = self.csv("c.json", json.dumps({"labels": ["a", "b"], "parser": {
            "answer_pattern": r"^Labels:\s*(.+?)\s*$", "case_insensitive": True,
            "multilabel_separator": ","}}))
        raw = self.csv("r.csv", 'item_id,config,raw_output\n1,c,"x\nLabels: B, a"\n'
                                '2,c,"Labels: none"\n3,c,"Labels: a, z"\n')
        out = str(self.dir / "o.csv")
        run(parse_outputs.main, cfg, raw, out)
        self.assertEqual(Path(out).read_text().split(), ["item_id,config,label", "1,c,a|b", "2,c,none", "3,c,"])


if __name__ == "__main__":
    unittest.main()
