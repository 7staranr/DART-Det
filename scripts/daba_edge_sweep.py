# -*- coding: utf-8 -*-
"""Sensitivity of the DABA gate to its tier edges (t1, t2).

The sweep must be run on a bucket where images actually change tier. On the
SKU-110K >=300 bucket the smallest proxy count at s>=0.1 is 262, above every
t2 candidate, so all 30 images land in the top tier under every setting and the
sweep compares one policy with itself. The 100-150 and 150-300 buckets are the
informative ones.

Reads only the per-image summary produced by wp1_eval.py, so it needs no
prediction cache and runs in under a second.

    python scripts/daba_edge_sweep.py --per-image results/.../sku_test_ft_per_image.csv
"""
import argparse
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EDGES = [(100, 200), (100, 250), (75, 200), (125, 200), (100, 150)]
BUCKETS = ["100-150", "150-300", ">=300"]


def main():
    ap = argparse.ArgumentParser()
    # default to the table this repo ships, so it runs against a clean clone
    _shipped = os.path.join(ROOT, "results", "per_image", "wp1_sku",
                            "sku_test_ft_per_image.csv")
    ap.add_argument("--per-image", default=_shipped if os.path.exists(_shipped)
                    else os.path.join(ROOT, "experiments", "wp1_sku",
                                      "sku_test_ft_per_image.csv"))
    ap.add_argument("--conf", type=float, default=0.1,
                    help="score threshold defining the density proxy n")
    ap.add_argument("--buckets", nargs="*", default=BUCKETS)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.per_image, encoding="utf-8")))
    col = f"ndet@conf{args.conf:g}"
    if col not in rows[0]:
        sys.exit(f"{col} not in {args.per_image}; columns: {list(rows[0])}")

    for bk in args.buckets:
        sel = [r for r in rows if r["bucket"] == bk]
        if not sel:
            print(f"\n=== bucket {bk}: no images ===")
            continue
        n = np.array([float(r[col]) for r in sel])
        n_gt = np.array([float(r["n_gt"]) for r in sel])
        matched = {k: np.array([float(r[f"matched@{k}"]) for r in sel])
                   for k in (300, 600, 1000)}

        print(f"\n=== bucket {bk}  ({len(sel)} images) ===")
        print(f"    density proxy at s>={args.conf:g}: "
              f"min {n.min():.0f}  mean {n.mean():.1f}  max {n.max():.0f}")
        for t1, t2 in EDGES:
            K = np.where(n < t1, 300, np.where(n < t2, 600, 1000))
            tiers = {v: int((K == v).sum()) for v in (300, 600, 1000)}
            recall = sum(matched[int(k)][i] for i, k in enumerate(K)) / n_gt.sum()
            flag = ("   [DEGENERATE: every image in one tier, "
                    "this setting is untestable here]"
                    if max(tiers.values()) == len(sel) else "")
            print(f"    t1={t1:3d} t2={t2:3d}: "
                  f"tiers {tiers[300]:5d}/{tiers[600]:5d}/{tiers[1000]:5d}  "
                  f"mean K={K.mean():6.1f}  recall={recall:.4f}{flag}")


if __name__ == "__main__":
    main()
