"""Step 5: statistical comparison.

1. Paired comparison of configurations on the same items: exact McNemar test
   on item-level correctness (multi-label: the whole label set must match),
   Holm correction across all pairs.
2. Optional F1 contrasts with a paired bootstrap (percentile 95% interval):
     --contrast classes   F1 differences between labels inside each configuration
                          (is label A really harder than label B?)
     --contrast configs   F1 differences between configurations, label by label
                          (did the new prompt really improve label A?)
   Resampling unit: items stratified by gold label (single-label data), plain
   items (multi-label data), or whole clusters with --clusters (several
   passages from one case, several posts from one account): items in one
   cluster are not independent, and resampling them one by one would make
   the intervals too narrow. All configurations are scored on the same
   resampled items. Intervals are pointwise, not adjusted for multiplicity.
3. A pre-registered decision rule for "difference" claims: |gap| > threshold
   AND the interval excludes 0. --threshold has no default: set it, before
   looking at results, to the smallest gap that would matter for your use.
   Otherwise report "no difference found", never "equivalent".

Usage: python scripts/compare.py gold.csv predictions.csv
           [--contrast none|classes|configs --threshold 0.10]
           [--clusters clusters.csv] [--boot 10000] [--seed 42]
"""
import argparse
import random
import sys
from collections import defaultdict
from itertools import combinations
from math import comb

from common import (UNPARSEABLE, check_coverage, is_multilabel, labels_of, load_gold, load_map,
                    load_predictions, macro_f1, per_label)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def holm(pvals):
    """pvals: {name: p}. Returns {name: adjusted p} (Holm step-down)."""
    order = sorted(pvals, key=pvals.get)
    m, adj, running = len(order), {}, 0.0
    for rank, name in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[name]))
        adj[name] = running
    return adj


def stratified_resample(items_by_class, rng):
    out = []
    for items in items_by_class.values():
        out.extend(rng.choice(items) for _ in items)
    return out


def cluster_resample(items_by_cluster, rng):
    keys = list(items_by_cluster)
    out = []
    for _ in keys:
        out.extend(items_by_cluster[rng.choice(keys)])
    return out


def percentile(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("gold")
    ap.add_argument("predictions")
    ap.add_argument("--contrast", choices=["none", "classes", "configs"], default="none")
    ap.add_argument("--threshold", type=float, help="required with --contrast; no default")
    ap.add_argument("--clusters", help="CSV with columns item_id,cluster: resample whole clusters")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    if a.contrast != "none" and a.threshold is None:
        ap.error("--threshold is required with --contrast: set the smallest F1 gap that "
                 "would matter for your use, before looking at the results")

    gold = load_gold(a.gold)
    preds = load_predictions(a.predictions)
    multi = is_multilabel(gold)
    labels = labels_of(gold.values())
    items = sorted(gold)
    for cfg in sorted(preds):
        check_coverage(cfg, preds[cfg], items)

    # 1. configurations: McNemar + Holm
    unit = " (correct = whole label set right)" if multi else ""
    print(f"== Paired configuration comparison (exact McNemar, Holm-corrected){unit}")
    raw, detail = {}, {}
    for x, y in combinations(sorted(preds), 2):
        cx = {i: preds[x].get(i, UNPARSEABLE) == gold[i] for i in items}
        cy = {i: preds[y].get(i, UNPARSEABLE) == gold[i] for i in items}
        b = sum(cx[i] and not cy[i] for i in items)
        c = sum(cy[i] and not cx[i] for i in items)
        name = f"{x} vs {y}"
        raw[name], detail[name] = mcnemar_exact(b, c), (b, c)
    adj = holm(raw)
    for name in raw:
        b, c = detail[name]
        print(f"  {name:<34} only-left {b:>3}  only-right {c:>3}  p {raw[name]:.4f}  Holm p {adj[name]:.4f}")
    if a.contrast == "none":
        return

    # 2. bootstrap: score every configuration on the same resampled items
    rng = random.Random(a.seed)
    if a.clusters:
        cmap = load_map(a.clusters, "item_id", "cluster")
        lone = [i for i in items if i not in cmap]
        if lone:
            print(f"WARNING: {len(lone)} items have no cluster; each is its own cluster", file=sys.stderr)
        groups = defaultdict(list)
        for i in items:
            groups[cmap.get(i, f"__{i}")].append(i)
        draw, how = (lambda: cluster_resample(groups, rng)), f"cluster bootstrap over {len(groups)} clusters"
    elif multi:
        draw, how = (lambda: [rng.choice(items) for _ in items]), "paired bootstrap"
    else:
        by_class = {c: [i for i in items if c in gold[i]] for c in labels}
        draw, how = (lambda: stratified_resample(by_class, rng)), "stratified paired bootstrap"

    point = {cfg: per_label([(gold[i], p.get(i, UNPARSEABLE)) for i in items], labels)
             for cfg, p in preds.items()}
    cfgs = sorted(preds)
    if a.contrast == "classes":
        keys = [(cfg, u, v) for cfg in cfgs for u, v in combinations(labels, 2)]
        diff = lambda f, k: f[k[0]][k[1]] - f[k[0]][k[2]]  # noqa: E731
    else:
        keys = [(x, y, c) for x, y in combinations(cfgs, 2) for c in labels]
        diff = lambda f, k: f[k[1]][k[2]] - f[k[0]][k[2]]  # noqa: E731
    draws = {k: [] for k in keys}
    for _ in range(a.boot):
        sample = draw()
        f1 = {cfg: {c: s["f1"] for c, s in per_label(
            [(gold[i], p.get(i, UNPARSEABLE)) for i in sample], labels).items()}
            for cfg, p in preds.items()}
        for k in keys:
            draws[k].append(diff(f1, k))
    pf1 = {cfg: {c: s["f1"] for c, s in sc.items()} for cfg, sc in point.items()}

    what = "Per-class F1 contrasts" if a.contrast == "classes" else "Per-class F1 changes between configurations"
    print(f"\n== {what} ({how}, B={a.boot}, seed={a.seed})")
    print(f"   decision rule: |gap| > {a.threshold} and 95% CI excludes 0")
    last = None
    for k in keys:
        if k[:-1 if a.contrast == "configs" else 1] != last:
            last = k[:-1 if a.contrast == "configs" else 1]
            if a.contrast == "classes":
                print(f"\n  {k[0]}  (macro-F1 {macro_f1(point[k[0]]):.3f})")
            else:
                print(f"\n  {k[0]} -> {k[1]}  (macro-F1 {macro_f1(point[k[0]]):.3f} -> {macro_f1(point[k[1]]):.3f})")
        d = diff(pf1, k)
        lo, hi = percentile(draws[k], 0.025), percentile(draws[k], 0.975)
        supported = abs(d) > a.threshold and (lo > 0 or hi < 0)
        verdict = "difference" if supported else "no difference found (not 'equivalent')"
        lhs = f"F1({k[1]}) - F1({k[2]})" if a.contrast == "classes" else f"F1({k[2]}) change"
        print(f"    {lhs} = {d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  -> {verdict}")


if __name__ == "__main__":
    main()
