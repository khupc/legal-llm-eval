"""Generate SYNTHETIC example data so every script can be tried end to end.
Nothing here is real annotation or model output.

Usage: python examples/make_toy_data.py
"""
import csv
import random
from pathlib import Path

HERE = Path(__file__).parent
LABELS = ["loan", "lease", "sale"]  # cause of action in civil judgments
rng = random.Random(7)


def noisy(true, p_correct):
    if rng.random() < p_correct:
        return true
    others = [l for l in LABELS if l != true]
    return rng.choice(others)


def write(name, header, rows):
    with open(HERE / name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


items = [(f"item{n:03d}", LABELS[n % 3]) for n in range(90)]
write("gold.csv", ["item_id", "label"], items)

ann = []
for i, g in items:
    for a in ("A1", "A2", "A3"):
        ann.append((i, a, noisy(g, 0.95)))
write("annotations.csv", ["item_id", "annotator", "label"], ann)

# per-class accuracy: 'lease' is the weak class in both configs
acc = {"zero_shot": {"lease": 0.55, "loan": 0.80, "sale": 0.82},
       "few_shot_cot": {"lease": 0.70, "loan": 0.92, "sale": 0.90}}
pred = []
for cfg, a in acc.items():
    for i, g in items:
        pred.append((i, cfg, noisy(g, a[g])))
write("predictions.csv", ["item_id", "config", "label"], pred)

mem = []
for i, g in items:
    for m, base in (("model_a", 0.80), ("model_b", 0.75), ("model_c", 0.70)):
        p = base - (0.2 if g == "lease" else 0)
        label = noisy(g, p)
        if m == "model_c" and rng.random() < 0.08:
            label = ""  # simulated degenerate / unparseable output
        mem.append((i, m, label))
write("members.csv", ["item_id", "member", "label"], mem)
print("wrote gold.csv, annotations.csv, predictions.csv, members.csv")
