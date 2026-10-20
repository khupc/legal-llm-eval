"""Step 6-7: ensemble disagreement and human-review routing.

Input  gold.csv and members.csv (item_id,member,label; empty = unparseable)
       where members are the models voting in an ensemble.
Output majority-vote accuracy, ties and splits by gold class, how often at
least one member was right in a tie (= a human reviewer could recover it),
unparseable and missing outputs per member, per-class majority-vote F1,
suggested routing rules, and a trade-off table (share sent to human review vs
accuracy) for increasingly strict routing levels.

A "split" is an item whose valid votes are not all the same (ties included).
Routing rules are triggered only by signals visible at run time (the votes);
gold is used to measure how well each rule would have worked.

Multi-label data ('a|b', 'none'): each label is voted on separately and is in
the majority set if more than half of the valid votes include it; a label
with exactly half the votes is a tie; a split is any label that some but not
all valid votes include. The members can also be N samples of one model
(self-consistency) instead of N different models.

Usage: python scripts/route.py examples/gold.csv examples/members.csv
"""
import sys
from collections import Counter, defaultdict

from common import UNPARSEABLE, is_multilabel, labels_of, load_gold, parse_cell, per_label, read_csv


def vote(valid, multilabel):
    """valid: list of label sets. Returns (majority set or UNPARSEABLE on a tie,
    contested labels: those the voters disagree on; empty = unanimous)."""
    if multilabel:
        n, cnt = len(valid), Counter(l for s in valid for l in s)
        contested = {l for l, k in cnt.items() if k < n}
        tie = not valid or any(2 * k == n for k in cnt.values())
        return (UNPARSEABLE if tie else frozenset(l for l, k in cnt.items() if 2 * k > n)), contested
    counts = Counter(valid).most_common()
    contested = set().union(*(s for s, _ in counts)) if len(counts) > 1 else set()
    tie = not counts or (len(counts) > 1 and counts[0][1] == counts[1][1])
    return (UNPARSEABLE if tie else counts[0][0]), contested


def main(gold_path, members_path, min_ratio=1.2):
    gold = load_gold(gold_path)
    multi = is_multilabel(gold)
    labels = labels_of(gold.values())
    votes = defaultdict(dict)
    for r in read_csv(members_path):
        votes[r["item_id"]][r["member"]] = parse_cell(r["label"])
    members = sorted({m for v in votes.values() for m in v})

    extra = sorted(set(votes) - set(gold))
    if extra:
        print(f"WARNING: {len(extra)} items in members file are not in gold, ignored, "
              f"e.g. {extra[:5]}", file=sys.stderr)

    unparse, missing = Counter(), Counter()
    ties, splits, flagged, correct, recoverable = [], [], [], 0, 0
    tie_gold, tie_voted, split_gold = Counter(), Counter(), Counter()
    majority, contested = {}, {}
    for i, g in gold.items():
        v = votes.get(i, {})
        missing.update(m for m in members if m not in v)
        unparse.update(m for m, l in v.items() if l is UNPARSEABLE)
        if len(v) < len(members) or UNPARSEABLE in v.values():
            flagged.append(i)
        valid = [l for l in v.values() if l is not UNPARSEABLE]
        majority[i], contested[i] = vote(valid, multi)
        if contested[i]:
            splits.append(i)
            split_gold.update(g)
        if majority[i] is UNPARSEABLE:      # a tie has no majority: counts as wrong
            ties.append(i)
            tie_gold.update(g)              # diagnosis: which true class causes ties
            tie_voted.update(sorted(set().union(*valid)))  # routing: labels visible at run time
            recoverable += g in valid
        else:
            correct += majority[i] == g
    if missing:
        print(f"WARNING: missing votes (no row at all, not unparseable): {dict(missing)}",
              file=sys.stderr)

    n, t = len(gold), len(ties) or 1
    w = max(14, max(len(c) for c in labels) + 2)
    f1 = {c: s["f1"] for c, s in per_label([(gold[i], majority[i]) for i in gold], labels).items()}
    print(f"members: {members}")
    exact = "exact label set, " if multi else ""
    print(f"majority-vote accuracy ({exact}ties count as wrong): {correct / n:.3f}")
    print(f"unparseable outputs per member: {dict(unparse) or 'none'}")
    print(f"missing votes per member: {dict(missing) or 'none'}")
    print(f"\nties: {len(ties)}/{n}; at least one member correct in {recoverable}/{len(ties)}")
    print("ties by true class (diagnosis):")
    for c, k in tie_gold.most_common():
        print(f"  {c:<{w}}{k:>3}  ({k / t:.0%} of ties)")
    print("labels voted in ties (what a router sees):")
    for c, k in tie_voted.most_common():
        print(f"  {c:<{w}}{k:>3}  ({k / t:.0%} of ties)")

    # per class: majority-vote F1, split rate among items of true class c relative
    # to the overall split rate (size-independent, so a large class does not look
    # "concentrated" just because it is large), and how the run-time rule
    # "split with a vote for c -> human" would have performed
    gold_n = Counter(l for g in gold.values() for l in g)
    overall = len(splits) / n
    ratio = {c: (split_gold[c] / gold_n[c]) / overall if overall else 0.0 for c in labels}
    print(f"\nsplits (not all valid votes agree, ties included): {len(splits)}/{n} ({overall:.0%})")
    print(f"{'class':<{w}}{'vote F1':>8}{'split rate':>12}{'vs all':>8}{'rule routes':>13}{'majority wrong':>16}")
    rule = {}
    for c in labels:
        routed = [i for i in splits if c in contested[i]]
        wrong = sum(majority[i] != gold[i] for i in routed)
        rule[c] = (len(routed), wrong)
        print(f"{c:<{w}}{f1[c]:>8.3f}{split_gold[c] / gold_n[c]:>12.0%}{ratio[c]:>7.2f}x{len(routed):>13}"
              f"{(f'{wrong}/{len(routed)}' if routed else '-'):>16}")

    print("\nsuggested routing rules:")
    print("  - any tie                -> human review")
    print("  - any unparseable member -> human review (log as reliability failure, do not re-run silently)")
    lowest = min(f1.values())
    for c in labels:
        if splits and f1[c] == lowest and ratio[c] >= min_ratio:
            k, w = rule[c]
            print(f"  - any vote for '{c}' that is not unanimous -> human review "
                  f"(lowest vote F1 {f1[c]:.3f}; split rate {ratio[c]:.2f}x the overall rate; "
                  f"routes {k} items, majority vote wrong in {w})")

    # cost/benefit of routing levels, each adding to the previous one
    weakest = [c for c in labels if f1[c] == lowest]
    levels = [
        ("machine only", set()),
        ("+ ties", set(ties)),
        ("+ unparseable/missing member", set(flagged)),
        (f"+ splits with a vote for {'/'.join(weakest)}",
         {i for i in splits if contested[i] & set(weakest)}),
        ("+ all splits (only unanimous auto-passed)", set(splits)),
    ]
    print("\nrouting trade-off (upper bound: assumes human review is always right):")
    print(f"  {'level':<44}{'human share':>12}{'auto acc':>10}{'overall acc':>13}{'errors left':>13}")
    routed = set()
    for name, add in levels:
        routed |= add
        auto = [i for i in gold if i not in routed]
        auto_ok = sum(majority[i] == gold[i] for i in auto)
        auto_acc = f"{auto_ok / len(auto):.3f}" if auto else "-"
        print(f"  {name:<44}{len(routed) / n:>11.0%} {auto_acc:>10}"
              f"{(len(routed) + auto_ok) / n:>13.3f}{len(auto) - auto_ok:>13}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
