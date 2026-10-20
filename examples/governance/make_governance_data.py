"""Generate a SYNTHETIC content-moderation example (no real platform data).

Scenario: LLM moderation of posts into one normal class and three violation
classes. The planted pattern: one class (hidden_aigc) has high precision but
low recall, and its misses mostly go to 'normal', i.e. violations that slip
through.

Usage: python examples/governance/make_governance_data.py
"""
import csv
import random
from pathlib import Path

HERE = Path(__file__).parent
LABELS = ["normal", "hidden_aigc", "false_claim", "off_platform"]
SIZES = {"normal": 180, "hidden_aigc": 40, "false_claim": 40, "off_platform": 40}
# where a wrong label goes, per gold class
DEST = {"normal": {"false_claim": 0.5, "off_platform": 0.3, "hidden_aigc": 0.2},
        "hidden_aigc": {"normal": 0.85, "false_claim": 0.15},
        "false_claim": {"normal": 0.7, "off_platform": 0.3},
        "off_platform": {"normal": 0.8, "false_claim": 0.2}}
rng = random.Random(2027)


def noisy(true, p_correct):
    if rng.random() < p_correct:
        return true
    dest = DEST[true]
    return rng.choices(list(dest), weights=list(dest.values()))[0]


def write(name, header, rows):
    with open(HERE / name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


items = []
for c, k in SIZES.items():
    items += [c] * k
rng.shuffle(items)
items = [(f"post{n:04d}", g) for n, g in enumerate(items)]
write("gold.csv", ["item_id", "label"], items)

ann = []
for i, g in items:
    for a in ("R1", "R2", "R3"):
        ann.append((i, a, noisy(g, 0.90 if g == "hidden_aigc" else 0.96)))
write("annotations.csv", ["item_id", "annotator", "label"], ann)

# prompt_v1: class names only; prompt_v2: adds a criterion and examples per class
acc = {"prompt_v1": {"normal": 0.93, "hidden_aigc": 0.45, "false_claim": 0.78, "off_platform": 0.88},
       "prompt_v2": {"normal": 0.95, "hidden_aigc": 0.60, "false_claim": 0.88, "off_platform": 0.92}}
pred = [(i, cfg, noisy(g, a[g])) for cfg, a in acc.items() for i, g in items]
write("predictions.csv", ["item_id", "config", "label"], pred)

members = {"model_a": {"normal": 0.94, "hidden_aigc": 0.55, "false_claim": 0.85, "off_platform": 0.90},
           "model_b": {"normal": 0.92, "hidden_aigc": 0.50, "false_claim": 0.82, "off_platform": 0.88},
           "model_c": {"normal": 0.90, "hidden_aigc": 0.45, "false_claim": 0.80, "off_platform": 0.85}}
mem = []
for i, g in items:
    for m, a in members.items():
        label = noisy(g, a[g])
        if m == "model_c" and rng.random() < 0.05:
            label = ""  # simulated unparseable output
        mem.append((i, m, label))
write("members.csv", ["item_id", "member", "label"], mem)
print(f"wrote {len(items)} synthetic posts to {HERE}")
