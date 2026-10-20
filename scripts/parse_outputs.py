"""Step 4-6: map raw model outputs to labels with the frozen parser rule.

Input  raw.csv: item_id,config,raw_output
Output predictions.csv: item_id,config,label   (label empty = unparseable)
Also prints unparseable counts per config and flags degenerate outputs
(very long runs of one repeated character, a typical decoding loop).

Multi-label: set parser.multilabel_separator (e.g. ",") and let the capture
group take the whole list, e.g. "^Labels:\\s*(.+?)\\s*$". Every listed label
must be a known label, otherwise the output is unparseable; "none" alone
means no label applies. Labels are written as 'a|b', the empty set as 'none'.

Usage: python scripts/parse_outputs.py config/frozen_config.json raw.csv predictions.csv
"""
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

from common import NONE, SEP, read_csv


def is_degenerate(text, run=50):
    return re.search(r"(.)\1{%d,}" % (run - 1), text, re.S) is not None


def main(config_path, raw_path, out_path):
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    flags = re.I if cfg["parser"].get("case_insensitive") else 0
    pattern = re.compile(cfg["parser"]["answer_pattern"], flags)
    labels = {l.lower(): l for l in cfg["labels"]}
    sep = cfg["parser"].get("multilabel_separator")

    def to_label(raw):
        if sep is None:
            return labels.get(raw.lower(), "")
        parts = [x.strip().lower() for x in raw.split(sep) if x.strip()]
        if parts == [NONE]:
            return NONE
        if not parts or any(x not in labels for x in parts):
            return ""
        return SEP.join(sorted({labels[x] for x in parts}))

    rows, unparse, degen = [], Counter(), Counter()
    for r in read_csv(raw_path):
        text = r["raw_output"].strip()
        last = text.splitlines()[-1] if text else ""
        m = pattern.match(last)
        label = to_label(m.group(1)) if m else ""
        if not label:
            unparse[r["config"]] += 1
            degen[r["config"]] += is_degenerate(text)
        rows.append((r["item_id"], r["config"], label))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "config", "label"])
        w.writerows(rows)
    for cfg_name in sorted({r[1] for r in rows}):
        print(f"{cfg_name}: unparseable {unparse[cfg_name]} (degenerate loops {degen[cfg_name]})")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
