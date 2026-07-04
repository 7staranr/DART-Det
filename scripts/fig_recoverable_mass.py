"""Empirical recoverable-mass profile b(n): the boundary-region size (recall
recoverable by widening the budget) versus per-image density, which the DABA
three-way cost model needs to be monotone increasing. From bucket CSVs."""
import csv
import os
ROOT = os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = os.path.join(ROOT, "experiments")
BUCKETS = ["<50", "50-100", "100-150", "150-300", ">=300"]
XPOS = [25, 75, 125, 225, 400]
SRC = [
    ("SKU-110K", r"wp1_sku\sku_test_ft_buckets.csv", "-o", "#1f77b4"),
    ("DOTA", r"wp1_ft\dota_ft_buckets.csv", "-s", "#2ca02c"),
    ("VisDrone test-dev", r"wp1_ft\visdrone_testdev_ft_s_buckets.csv", "-^", "#ff7f0e"),
]


def load(path):
    d = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            d[r["bucket"].strip()] = r
    return d


fig, ax = plt.subplots(figsize=(6.2, 4.2))
for name, fn, st, c in SRC:
    d = load(os.path.join(EXP, fn))
    xs, ys = [], []
    for b, x in zip(BUCKETS, XPOS):
        if b in d and int(d[b]["n_images"]) >= 3:
            r3, r10 = float(d[b]["R@300"]), float(d[b]["R@1000"])
            xs.append(x)
            ys.append(r10 - r3)  # boundary mass = recoverable recall
    ax.plot(xs, ys, st, label=name, color=c, lw=1.8, ms=6)
ax.set_xlabel("per-image object count (density)")
ax.set_ylabel(r"recoverable mass $b(n)=R@1000-R@300$")
ax.set_title("Recoverable boundary mass rises with density")
ax.axhline(0, color="gray", lw=0.8)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
out = os.path.join(ROOT, "paper", "figures", "fig_recoverable_mass.pdf")
plt.savefig(out, bbox_inches="tight")
print("wrote", out)
for name, fn, _, _ in SRC:
    d = load(os.path.join(EXP, fn))
    print(name, [(b, round(float(d[b]["R@1000"]) - float(d[b]["R@300"]), 3))
                 for b in BUCKETS if b in d and int(d[b]["n_images"]) >= 3])
