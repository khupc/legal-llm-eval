"""Generate a SYNTHETIC multi-label example (no real contracts).

Scenario: an LLM tags contract clauses with risk types. A clause can carry
several risks at once, or none. Clauses come in groups of five from the same
contract, and some contracts are harder than others, so clauses are not
independent: use clusters.csv with compare.py --clusters.
Planted pattern: 'unilateral_change' is often missed.

Usage: python examples/multilabel/make_multilabel_data.py
"""
import csv
import random
from pathlib import Path

HERE = Path(__file__).parent
LABELS = ["liability_cap", "auto_renewal", "unilateral_change"]
rng = random.Random(11)


def cell(s):
    return "|".join(sorted(s)) or "none"


def write(name, header, rows):
    with open(HERE / name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


items, clusters, hard = [], [], set()
for k in range(40):
    contract = f"contract{k:02d}"
    if rng.random() < 0.25:
        hard.add(contract)
    for j in range(5):
        i = f"{contract}-c{j}"
        items.append((i, {l for l in LABELS if rng.random() < 0.3}))
        clusters.append((i, contract))
write("gold.csv", ["item_id", "label"], [(i, cell(g)) for i, g in items])
write("clusters.csv", ["item_id", "cluster"], clusters)
contract_of = dict(clusters)


def predict(i, g, recall, fp=0.03):
    r = 0.7 if contract_of[i] in hard else 1.0
    return {l for l in LABELS if (rng.random() < recall[l] * r if l in g else rng.random() < fp)}


ann = []
for i, g in items:
    for a in ("R1", "R2", "R3"):
        ann.append((i, a, cell({l for l in LABELS if (l in g) != (rng.random() < 0.04)})))
write("annotations.csv", ["item_id", "annotator", "label"], ann)

acc = {"prompt_v1": {"liability_cap": 0.90, "auto_renewal": 0.88, "unilateral_change": 0.50},
       "prompt_v2": {"liability_cap": 0.92, "auto_renewal": 0.90, "unilateral_change": 0.65}}
write("predictions.csv", ["item_id", "config", "label"],
      [(i, cfg, cell(predict(i, g, a))) for cfg, a in acc.items() for i, g in items])

members = {"model_a": {"liability_cap": 0.90, "auto_renewal": 0.90, "unilateral_change": 0.55},
           "model_b": {"liability_cap": 0.88, "auto_renewal": 0.85, "unilateral_change": 0.50},
           "model_c": {"liability_cap": 0.85, "auto_renewal": 0.85, "unilateral_change": 0.45}}
mem = []
for i, g in items:
    for m, a in members.items():
        label = cell(predict(i, g, a))
        if m == "model_c" and rng.random() < 0.04:
            label = ""  # simulated unparseable output
        mem.append((i, m, label))
write("members.csv", ["item_id", "member", "label"], mem)
print(f"wrote {len(items)} synthetic clauses from 40 contracts to {HERE}")
