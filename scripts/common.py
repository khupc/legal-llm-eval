"""Shared helpers: CSV loading and per-label metrics. Standard library only.

Every label cell is read as a SET of labels, so single-label and multi-label
tasks share one code path:
  "loan"                       -> {"loan"}
  "false_claim|off_platform"   -> {"false_claim", "off_platform"}  (multi-label)
  "none"                       -> set()        (multi-label: no label applies)
  ""                           -> UNPARSEABLE  (the output could not be parsed)
A task is multi-label as soon as any gold item has zero or several labels.
"""
import csv
import sys
from collections import Counter, defaultdict

UNPARSEABLE = None
SEP = "|"
NONE = "none"


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_cell(text):
    t = (text or "").strip()
    if not t:
        return UNPARSEABLE
    if t == NONE:
        return frozenset()
    return frozenset(x.strip() for x in t.split(SEP) if x.strip())


def as_set(x):
    return x if x is UNPARSEABLE or isinstance(x, (set, frozenset)) else parse_cell(x)


def fmt(s):
    return "" if s is UNPARSEABLE else (SEP.join(sorted(s)) or NONE)


def load_gold(path):
    """gold.csv: item_id,label  ->  {item_id: frozenset of labels}"""
    gold = {r["item_id"]: parse_cell(r["label"]) for r in read_csv(path)}
    bad = [i for i, s in gold.items() if s is UNPARSEABLE]
    if bad:
        sys.exit(f"gold has empty labels (use '{NONE}' for 'no label applies'): {bad[:5]}")
    return gold


def load_predictions(path):
    """predictions.csv: item_id,config,label  (label empty = unparseable)
    Returns {config: {item_id: frozenset or UNPARSEABLE}}."""
    preds = defaultdict(dict)
    for r in read_csv(path):
        preds[r["config"]][r["item_id"]] = parse_cell(r["label"])
    return dict(preds)


def load_map(path, key, value):
    """Two-column CSV -> {key: value}; used for label groups and item clusters."""
    return {r[key]: r[value] for r in read_csv(path)}


def is_multilabel(gold):
    return any(len(s) != 1 for s in gold.values())


def labels_of(sets):
    return sorted(set().union(*sets))


def coarsen(s, groups):
    """Map fine labels to their group (labels without a group keep their name)."""
    return s if s is UNPARSEABLE else frozenset(groups.get(l, l) for l in s)


def check_coverage(cfg, pred, items):
    """Return gold items that have no prediction row at all for `cfg`.
    A missing row is a pipeline failure (item not run), not an unparseable
    output, so it is reported separately and warned about on stderr."""
    missing = [i for i in items if i not in pred]
    if missing:
        print(f"WARNING: {cfg}: {len(missing)} gold items have no prediction row "
              f"(not run?), e.g. {missing[:5]}; scored as wrong", file=sys.stderr)
    return missing


def _counts(pairs):
    tp, fp, fn = Counter(), Counter(), Counter()
    for g, p in pairs:
        if p is UNPARSEABLE:
            fn.update(g)
            continue
        tp.update(g & p)
        fn.update(g - p)
        fp.update(p - g)
    return tp, fp, fn


def _prf(tp, fp, fn):
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def per_label(pairs, labels):
    """Precision, recall, F1 per label over (gold set, predicted set) pairs.
    Unparseable predictions count as missed for every gold label (recall) and
    as a prediction of nothing (precision): a strict denominator."""
    tp, fp, fn = _counts(pairs)
    return {c: _prf(tp[c], fp[c], fn[c]) for c in labels}


def micro_f1(pairs):
    tp, fp, fn = (sum(c.values()) for c in _counts(pairs))
    return _prf(tp, fp, fn)["f1"]


def per_class(gold, pred, labels, items):
    """Single-label convenience wrapper: dicts of labels (str or set) over `items`."""
    return per_label([(as_set(gold[i]), as_set(pred.get(i, UNPARSEABLE))) for i in items], labels)


def macro_f1(scores):
    return sum(s["f1"] for s in scores.values()) / len(scores)
