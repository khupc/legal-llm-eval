"""Regression tests for input-robustness fixes (missing rows, paths, duplicates).
Run: python -m unittest discover -s tests
"""
import io
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import agreement  # noqa: E402
import evaluate  # noqa: E402
import freeze  # noqa: E402
import route  # noqa: E402


def run(fn, *args):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        fn(*args)
    return out.getvalue(), err.getvalue()


class TmpDir(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.dir)

    def csv(self, name, text):
        p = self.dir / name
        p.write_text(text, encoding="utf-8")
        return str(p)


class TestEvaluateMissing(TmpDir):
    def test_missing_row_is_not_unparseable(self):
        gold = self.csv("g.csv", "item_id,label\n1,a\n2,a\n3,b\n")
        pred = self.csv("p.csv", "item_id,config,label\n1,c,a\n2,c,\n")  # 2 unparseable, 3 missing
        out, err = run(evaluate.main, [gold, pred])
        self.assertIn("unparseable 1/3", out)
        self.assertIn("missing 1/3", out)
        self.assertIn("WARNING", err)


class TestFreezePaths(TmpDir):
    def test_verify_accepts_equivalent_paths(self):
        (self.dir / "config" / "prompts").mkdir(parents=True)
        (self.dir / "config" / "prompts" / "p.txt").write_text("prompt", encoding="utf-8")
        (self.dir / "config" / "c.json").write_text(
            '{"prompts": {"p": "config/prompts/p.txt"}}', encoding="utf-8")
        cwd = os.getcwd()
        os.chdir(self.dir)
        try:
            run(freeze.main, "lock", "config/c.json")
            for spelling in ("./config/c.json", str(self.dir / "config" / "c.json")):
                out, _ = run(freeze.main, "verify", spelling)
                self.assertIn("unchanged", out)
            (self.dir / "config" / "prompts" / "p.txt").write_text("edited", encoding="utf-8")
            with self.assertRaises(SystemExit):
                run(freeze.main, "verify", "./config/c.json")
        finally:
            os.chdir(cwd)


class TestRouteCoverage(TmpDir):
    def test_gold_is_the_denominator(self):
        gold = self.csv("g.csv", "item_id,label\n1,a\n2,b\n")
        # item 2 has no votes at all; item 9 is not in gold; m2 never voted on 1
        mem = self.csv("m.csv", "item_id,member,label\n1,m1,a\n9,m1,a\n9,m2,b\n")
        out, err = run(route.main, gold, mem)
        self.assertIn("majority-vote accuracy (ties count as wrong): 0.500", out)
        self.assertIn("unparseable outputs per member: none", out)
        self.assertIn("missing votes per member: {'m2': 2, 'm1': 1}", out)
        self.assertIn("not in gold", err)


class TestRouteTradeoff(TmpDir):
    def test_unanimous_wrong_is_never_routed(self):
        gold = self.csv("g.csv", "item_id,label\n1,a\n2,a\n3,a\n4,a\n")
        rows = ["item_id,member,label"]
        for i, labs in {1: "aaa", 2: "aab", 3: "abc", 4: "bbb"}.items():
            rows += [f"{i},m{k},{l}" for k, l in enumerate(labs)]
        mem = self.csv("m.csv", "\n".join(rows) + "\n")
        out, _ = run(route.main, gold, mem)
        lines = {l.split("  ")[1].strip(): l.split() for l in out.splitlines()
                 if l.startswith("  machine") or l.startswith("  + ")}
        self.assertEqual(lines["machine only"][-3:], ["0.500", "0.500", "2"])
        self.assertEqual(lines["+ ties"][-4:], ["25%", "0.667", "0.750", "1"])
        # item 4 is unanimous but wrong: even routing every split leaves 1 error
        self.assertEqual(lines["+ all splits (only unanimous auto-passed)"][-1], "1")


class TestAgreementIncomplete(TmpDir):
    def test_missing_annotation_does_not_abort(self):
        ann = self.csv("a.csv", "item_id,annotator,label\n"
                                "1,A,x\n1,B,x\n2,A,x\n2,B,y\n3,A,y\n")  # item 3 lacks B
        out, _ = run(agreement.main, ann)
        self.assertIn("Fleiss' kappa: n/a", out)
        self.assertIn("Cohen's kappa A vs B", out)
        self.assertIn("raw agreement (all raters identical): 0.500", out)  # item 3 excluded
        self.assertIn("conflicts to adjudicate (1)", out)

    def test_duplicate_row_warns(self):
        ann = self.csv("a.csv", "item_id,annotator,label\n1,A,x\n1,A,y\n1,B,y\n")
        _, err = run(agreement.main, ann)
        self.assertIn("duplicate", err)

    def test_no_shared_items(self):
        ann = self.csv("a.csv", "item_id,annotator,label\n1,A,x\n2,B,x\n")
        out, _ = run(agreement.main, ann)
        self.assertIn("n/a (no shared items)", out)


if __name__ == "__main__":
    unittest.main()
