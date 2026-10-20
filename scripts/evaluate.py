"""Step 5: scores per configuration.

Input  gold.csv (item_id,label) and predictions.csv (item_id,config,label;
       empty label = unparseable output; 'a|b' = several labels; 'none' = no label)
Output per config: macro-F1, per-label P/R/F1, unparseable count, missing
count (gold items with no prediction row: a run failure, kept apart from
unparseable), and the share of items carrying each label, predicted vs gold
(to spot a label the model systematically under-predicts).
Multi-label data also get micro-F1, exact-match rate (whole label set right)
and labels per item, predicted vs gold (to spot systematic under-labelling).

--groups labels.csv (label,group) adds the same report at group level, e.g.
fine violation types -> violation / normal, which gives the miss rate and the
false-alarm rate of the coarse decision.

Usage: python scripts/evaluate.py gold.csv predictions.csv [--groups groups.csv]
"""
import argparse

from common import (UNPARSEABLE, check_coverage, coarsen, is_multilabel, labels_of, load_gold,
                    load_map, load_predictions, macro_f1, micro_f1, per_label)


def show(name, gold, pred, items, multi, n_missing):
    n = len(items)
    pairs = [(gold[i], pred.get(i, UNPARSEABLE)) for i in items]
    labels = labels_of(gold[i] for i in items)
    scores = per_label(pairs, labels)
    unparse = sum(i in pred and pred[i] is UNPARSEABLE for i in items)
    parsed = [i for i in items if pred.get(i, UNPARSEABLE) is not UNPARSEABLE]
    n_parsed = len(parsed) or 1
    head = f"\n== {name}   macro-F1 {macro_f1(scores):.3f}"
    if multi:
        exact = sum(g == p for g, p in pairs) / n
        head += f"   micro-F1 {micro_f1(pairs):.3f}   exact match {exact:.3f}"
    print(head + f"   unparseable {unparse}/{n}   missing {n_missing}/{n}")
    if multi:
        per_pred = sum(len(pred[i]) for i in parsed) / n_parsed
        per_gold = sum(len(gold[i]) for i in parsed) / n_parsed
        print(f"labels per item (parsed items): predicted {per_pred:.2f} vs gold {per_gold:.2f}")
    w = max(14, max(len(c) for c in labels) + 2)
    print(f"{'class':<{w}}{'P':>7}{'R':>7}{'F1':>7}{'pred share':>12}{'gold share':>12}")
    for c in labels:
        s = scores[c]
        ps = sum(c in pred[i] for i in parsed) / n_parsed
        gs = sum(c in gold[i] for i in items) / n
        flag = "  <- under-predicted" if ps < gs - 0.05 else ""
        print(f"{c:<{w}}{s['precision']:>7.3f}{s['recall']:>7.3f}{s['f1']:>7.3f}{ps:>12.2f}{gs:>12.2f}{flag}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("gold")
    ap.add_argument("predictions")
    ap.add_argument("--groups", help="CSV with columns label,group for a coarser second report")
    a = ap.parse_args(argv)

    gold = load_gold(a.gold)
    preds = load_predictions(a.predictions)
    multi = is_multilabel(gold)
    items = sorted(gold)
    groups = load_map(a.groups, "label", "group") if a.groups else None
    print(f"mode: {'multi-label' if multi else 'single-label'}")

    for cfg, pred in sorted(preds.items()):
        missing = check_coverage(cfg, pred, items)
        show(cfg, gold, pred, items, multi, len(missing))
        if groups:
            show(f"{cfg} [group level]", {i: coarsen(gold[i], groups) for i in items},
                 {i: coarsen(p, groups) for i, p in pred.items()}, items, multi, len(missing))


if __name__ == "__main__":
    main()
