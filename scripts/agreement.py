"""Step 3: inter-annotator agreement.

Input  annotations.csv: item_id,annotator,label
       ('a|b' = several labels, 'none' = no label, empty = annotator abstained)
Output raw agreement, Fleiss' kappa, pairwise Cohen's kappa, Fleiss' kappa
per label (that label vs all others), and the list of items where annotators
disagree (to send to an independent adjudicator).
Multi-label data: raw agreement means identical label sets; kappa is reported
per label only (presence vs absence of each label).

Usage: python scripts/agreement.py examples/annotations.csv
"""
import sys
from collections import Counter, defaultdict
from itertools import combinations

from common import UNPARSEABLE, fmt, labels_of, parse_cell, read_csv


def fleiss_kappa(table, labels):
    """table: {item: Counter(label -> n_raters)}; every item needs the same n."""
    n = {sum(c.values()) for c in table.values()}
    if len(n) != 1:
        raise ValueError("Fleiss' kappa needs the same number of raters per item")
    n = n.pop()
    if n < 2:
        raise ValueError("Fleiss' kappa needs at least 2 raters per item")
    N = len(table)
    p_j = {c: sum(t[c] for t in table.values()) / (N * n) for c in labels}
    P_i = [(sum(v * v for v in t.values()) - n) / (n * (n - 1)) for t in table.values()]
    P_bar = sum(P_i) / N
    P_e = sum(p * p for p in p_j.values())
    return (P_bar - P_e) / (1 - P_e) if P_e < 1 else 1.0


def cohen_kappa(a, b):
    items = sorted(set(a) & set(b))
    po = sum(a[i] == b[i] for i in items) / len(items)
    ca, cb = Counter(a[i] for i in items), Counter(b[i] for i in items)
    pe = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / len(items) ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main(path):
    rows = read_csv(path)
    by_item = defaultdict(dict)
    dupes, abstained = [], 0
    for r in rows:
        i, who, s = r["item_id"], r["annotator"], parse_cell(r["label"])
        if who in by_item[i]:
            dupes.append((i, who))
        if s is UNPARSEABLE:
            abstained += 1
            continue
        by_item[i][who] = s
    if dupes:
        print(f"WARNING: {len(dupes)} duplicate (item_id, annotator) rows; the last one was kept, "
              f"e.g. {dupes[:5]}", file=sys.stderr)
    by_item = {i: v for i, v in by_item.items() if v}
    labels = labels_of(s for v in by_item.values() for s in v.values())
    annotators = sorted({r["annotator"] for r in rows})
    multilabel = any(len(s) != 1 for v in by_item.values() for s in v.values())
    if not multilabel:  # single-label: work with the plain label strings
        by_item = {i: {a: next(iter(s)) for a, s in v.items()} for i, v in by_item.items()}

    table = {i: Counter(v.values()) for i, v in by_item.items()}
    rated2 = [i for i, v in by_item.items() if len(v) >= 2]
    unanimous = [i for i in rated2 if len(table[i]) == 1]
    conflicts = [i for i, c in table.items() if len(c) > 1]

    mode = "  (multi-label)" if multilabel else ""
    print(f"items: {len(table)}  annotators: {len(annotators)}  labels: {labels}{mode}")
    if abstained:
        print(f"abstentions (empty label, treated as not annotated): {abstained}")
    incomplete = sum(len(v) < len(annotators) for v in by_item.values())
    if incomplete:
        print(f"items missing at least one annotator: {incomplete}  "
              f"(single-annotator items: {len(table) - len(rated2)}, excluded from raw agreement)")
    if rated2:
        what = "identical label sets" if multilabel else "all raters identical"
        print(f"raw agreement ({what}): {len(unanimous) / len(rated2):.3f}")
    if not multilabel:
        try:
            print(f"Fleiss' kappa: {fleiss_kappa(table, labels):.3f}")
        except ValueError as e:
            print(f"Fleiss' kappa: n/a ({e})")
        for x, y in combinations(annotators, 2):
            a = {i: v[x] for i, v in by_item.items() if x in v}
            b = {i: v[y] for i, v in by_item.items() if y in v}
            if set(a) & set(b):
                print(f"Cohen's kappa {x} vs {y}: {cohen_kappa(a, b):.3f}")
            else:
                print(f"Cohen's kappa {x} vs {y}: n/a (no shared items)")
    print("Fleiss' kappa per label (label vs all others):")
    w = max(14, max(len(c) for c in labels) + 2)
    for c in labels:
        has = (lambda s: c in s) if multilabel else (lambda s: s == c)
        t = {i: Counter({True: sum(map(has, v.values())), False: sum(not has(s) for s in v.values())})
             for i, v in by_item.items()}
        try:
            print(f"  {c:<{w}}{fleiss_kappa(t, [True, False]):.3f}")
        except ValueError as e:
            print(f"  {c:<{w}}n/a ({e})")
    print(f"\nconflicts to adjudicate ({len(conflicts)}):")
    for i in conflicts:
        print("  ", i, {a: fmt(s) for a, s in by_item[i].items()} if multilabel else dict(by_item[i]))


if __name__ == "__main__":
    main(sys.argv[1])
