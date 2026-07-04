"""WP1 plots: recall-density curves and budget saturation, from wp1_eval outputs.

Produces (per dataset):
  fig1: AR@k vs density bucket, one line per k (100/300/600/1000), per model.
  fig2: per-image scatter -- n_gt vs recall@300, with saturation marked.
  fig3: budget saturation rate vs bucket, per deployment conf threshold.

Usage:
  python wp1_plot.py --per-image <csv...> --labels <name...> --out-dir <dir> --title <t>
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BUCKET_NAMES = ["<50", "50-100", "100-150", "150-300", ">300"]
KS = [100, 300, 600, 1000]


def load_per_image(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def bucket_ar(rows, k):
    out = {}
    by_b = defaultdict(lambda: [0, 0])
    for r in rows:
        b = r["bucket"]
        by_b[b][0] += int(r[f"matched@{k}"])
        by_b[b][1] += int(r["n_gt"])
    for b, (m, g) in by_b.items():
        out[b] = m / g if g else float("nan")
    return out


def bucket_sat(rows, conf):
    out = {}
    by_b = defaultdict(lambda: [0, 0])
    for r in rows:
        b = r["bucket"]
        by_b[b][0] += int(r[f"sat@conf{conf}"])
        by_b[b][1] += 1
    for b, (s, n) in by_b.items():
        out[b] = s / n if n else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-image", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    datasets = [(lab, load_per_image(p))
                for lab, p in zip(args.labels, args.per_image)]

    # fig1: AR@k vs bucket per model
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4),
                             squeeze=False)
    for ax, (lab, rows) in zip(axes[0], datasets):
        x = np.arange(len(BUCKET_NAMES))
        present = [b for b in BUCKET_NAMES
                   if any(r["bucket"] == b for r in rows)]
        for k in KS:
            ar = bucket_ar(rows, k)
            ax.plot([BUCKET_NAMES.index(b) for b in present],
                    [ar.get(b, np.nan) for b in present],
                    marker="o", label=f"AR@{k}")
        ax.set_xticks(x)
        ax.set_xticklabels(BUCKET_NAMES)
        ax.set_xlabel("GT objects per image")
        ax.set_ylabel("Recall (IoU 0.5, class-agnostic)")
        ax.set_title(lab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(args.title)
    fig.tight_layout()
    p1 = os.path.join(args.out_dir, "fig1_recall_density.png")
    fig.savefig(p1, dpi=160)
    print("wrote", p1)

    # fig2: scatter n_gt vs recall@300
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4),
                             squeeze=False)
    for ax, (lab, rows) in zip(axes[0], datasets):
        ngt = np.array([int(r["n_gt"]) for r in rows])
        rec = np.array([int(r["matched@300"]) / max(int(r["n_gt"]), 1)
                        for r in rows])
        sat = np.array([int(r["sat@conf0.25"]) for r in rows], dtype=bool)
        ax.scatter(ngt[~sat], rec[~sat], s=8, alpha=0.4, label="not saturated")
        if sat.any():
            ax.scatter(ngt[sat], rec[sat], s=12, alpha=0.7, color="red",
                       label="ndet@0.25 >= 300")
        ax.axvline(300, color="gray", ls="--", lw=1)
        ax.set_xlabel("GT objects per image")
        ax.set_ylabel("recall@300 (IoU 0.5)")
        ax.set_title(lab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(args.title)
    fig.tight_layout()
    p2 = os.path.join(args.out_dir, "fig2_scatter_recall.png")
    fig.savefig(p2, dpi=160)
    print("wrote", p2)

    # fig3: saturation rate per bucket
    fig, ax = plt.subplots(figsize=(6.4, 4))
    width = 0.35
    x = np.arange(len(BUCKET_NAMES))
    for i, (lab, rows) in enumerate(datasets):
        sat25 = bucket_sat(rows, 0.25)
        vals = [sat25.get(b, 0) for b in BUCKET_NAMES]
        ax.bar(x + (i - len(datasets) / 2 + 0.5) * width / len(datasets) * 2,
               vals, width / len(datasets) * 2, label=f"{lab} conf>=0.25")
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKET_NAMES)
    ax.set_ylabel("fraction of images with ndet >= 300")
    ax.set_title("Deployment budget saturation " + args.title)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p3 = os.path.join(args.out_dir, "fig3_saturation.png")
    fig.savefig(p3, dpi=160)
    print("wrote", p3)


if __name__ == "__main__":
    main()
