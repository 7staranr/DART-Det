"""Generate paper figures from final results. Values either read from the
bucket CSVs (recall-density) or taken from the verified findings in
verified experiment findings (slot composition, attribution, AP with CI, mask intervention).
Outputs vector PDFs to paper/figures/.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150})
BN = ["<50", "50-100", "100-150", "150-300", ">=300"]
BX = ["<50", "50-\n100", "100-\n150", "150-\n300", "≥300"]


def load_buckets(path):
    d = {r["bucket"]: r for r in csv.DictReader(open(path, encoding="utf-8"))}
    return d


# ---- Fig 2: recall-density curves (finetuned weights) ----
def fig_recall_density():
    srcs = [
        ("VisDrone yolo26n", os.path.join(ROOT, "experiments", "wp1_ft",
         "visdrone_ft_n_buckets.csv")),
        ("VisDrone yolo26s", os.path.join(ROOT, "experiments", "wp1_ft",
         "visdrone_ft_s_buckets.csv")),
        ("SKU-110K yolo26n", os.path.join(ROOT, "experiments", "wp1_sku",
         "sku_test_ft_buckets.csv")),
    ]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for name, p in srcs:
        d = load_buckets(p)
        x = [i for i, b in enumerate(BN) if b in d]
        y = [float(d[BN[i]]["R@300"]) for i in x]
        ax.plot(x, y, marker="o", label=name)
    ax.set_xticks(range(len(BN)))
    ax.set_xticklabels(BX)
    ax.set_xlabel("ground-truth objects per image (density bucket)")
    ax.set_ylabel("Recall@300 (IoU 0.5)")
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=8)
    ax.set_title("Recall collapses as scene density rises")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_recall_density.pdf"))
    print("wrote fig_recall_density.pdf")


# ---- Fig 3: TP/duplicate/FP top-300 slot composition ----
def fig_slot_composition():
    # %TP-distinct / %dup / %FP  (SKU)
    sku = {"<50": (9.4, 4.6, 86.0), "50-100": (26.5, 8.5, 65.1),
           "100-150": (41.8, 12.0, 46.2), "150-300": (57.0, 12.2, 30.8),
           ">=300": (88.4, 2.6, 9.0)}
    tp = [sku[b][0] for b in BN]
    dup = [sku[b][1] for b in BN]
    fp = [sku[b][2] for b in BN]
    x = np.arange(len(BN))
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.bar(x, tp, label="TP distinct (hard exhaustion)",
           color="#2c7fb8")
    ax.bar(x, dup, bottom=tp, label="duplicate", color="#7fcdbb")
    ax.bar(x, fp, bottom=np.array(tp) + np.array(dup),
           label="FP (rank displacement)", color="#edf8b1")
    ax.set_xticks(x); ax.set_xticklabels(BX)
    ax.set_xlabel("density bucket")
    ax.set_ylabel("top-300 slot composition (%)")
    ax.set_title("From rank displacement to hard exhaustion")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_slot_composition.pdf"))
    print("wrote fig_slot_composition.pdf")


# ---- Fig 4: attribution decomposition ----
def fig_attribution():
    # ft_s / ft_n shares (%)
    labels = ["budget\ntruncation", "shared dense\ndifficulty",
              "o2o-head\nresidual"]
    ft_s = [56, 34, 10]
    ft_n = [51, 35, 14]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.bar(x - w / 2, ft_s, w, label="yolo26s", color="#2c7fb8")
    ax.bar(x + w / 2, ft_n, w, label="yolo26n", color="#a1dab4")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("share of dense recall loss (%)")
    ax.set_title("Attribution of the dense recall collapse (VisDrone)")
    ax.legend(fontsize=8)
    ax.grid(False)
    for i, (a, b) in enumerate(zip(ft_s, ft_n)):
        ax.text(i - w / 2, a + 1, f"{a}%", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 1, f"{b}%", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_attribution.pdf"))
    print("wrote fig_attribution.pdf")


# ---- Fig 5: AP vs budget per bucket + dense-subset CI ----
def fig_ap_budget():
    # SKU per-bucket mAP50 at 300/600/1000 (from wp4_ap_per_bucket run)
    sku = {"50-100": (0.7894, 0.7931, 0.7942),
           "100-150": (0.9455, None, None)}  # 100-150 only had @300 logged
    # dense-subset (GT>=150) point + CI (custom all-point AP)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    # left: SKU-dense AP@k with bootstrap CI (0.888/0.911/0.915)
    ks = [300, 600, 1000]
    ap = [0.8880, 0.9111, 0.9150]
    lo = [0.8802, 0.9046, 0.9090]
    hi = [0.8959, 0.9171, 0.9206]
    err = [np.array(ap) - np.array(lo), np.array(hi) - np.array(ap)]
    axes[0].errorbar(ks, ap, yerr=err, marker="o", capsize=4, color="#2c7fb8")
    axes[0].set_xlabel("decode budget $K$")
    axes[0].set_ylabel("AP@0.5 (all-point)")
    axes[0].set_title("SKU-110K dense subset (n=1034)\n+2.7 AP, 95% CI [+2.3,+3.1]")
    axes[0].set_xticks(ks)
    # right: precision flat (val operating point) SKU-dense 0.899->0.898
    axes[1].plot(ks, [0.8987, 0.8981, 0.8981], marker="s", color="#d95f0e",
                 label="precision (flat)")
    axes[1].plot(ks, [0.844, 0.848, 0.848], marker="^", color="#2c7fb8",
                 label="recall")
    axes[1].set_xlabel("decode budget $K$")
    axes[1].set_ylabel("validator metric")
    axes[1].set_title("Precision neutral as budget rises")
    axes[1].set_xticks(ks); axes[1].set_ylim(0.8, 0.95)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_ap_budget.pdf"))
    print("wrote fig_ap_budget.pdf")


# ---- Fig 6: mask intervention newly-detectable rate ----
def fig_mask():
    thr = [0.05, 0.10, 0.25]
    dense = [24.2, 16.2, 6.2]
    dlo = [17.5, 10.8, 2.5]
    dhi = [31.7, 22.1, 10.8]
    placebo = [0.0, 0.0, 0.0]
    x = np.arange(len(thr))
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    err = [np.array(dense) - np.array(dlo), np.array(dhi) - np.array(dense)]
    ax.bar(x - 0.2, dense, 0.4, yerr=err, capsize=4,
           label="dense (competitors removed)", color="#2c7fb8")
    ax.bar(x + 0.2, placebo, 0.4, label="placebo (sparse, same mask)",
           color="#bdbdbd")
    ax.set_xticks(x); ax.set_xticklabels([f"conf$\\geq${t}" for t in thr])
    ax.set_ylabel("missed dense objects made\ndetectable by masking (%)")
    ax.set_title("Competitors causally suppress detection")
    ax.legend(fontsize=8)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_mask.pdf"))
    print("wrote fig_mask.pdf")


def fig_teaser():
    """2-panel teaser: recall-density decline + slot-composition transition."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 3.3))
    srcs = [
        ("VisDrone -n", os.path.join(ROOT, "experiments", "wp1_ft",
         "visdrone_ft_n_buckets.csv")),
        ("VisDrone -s", os.path.join(ROOT, "experiments", "wp1_ft",
         "visdrone_ft_s_buckets.csv")),
        ("SKU-110K -n", os.path.join(ROOT, "experiments", "wp1_sku",
         "sku_test_ft_buckets.csv")),
    ]
    for name, p in srcs:
        d = load_buckets(p)
        xs = [i for i, bk in enumerate(BN) if bk in d]
        a.plot(xs, [float(d[BN[i]]["R@300"]) for i in xs], marker="o",
               label=name)
    a.set_xticks(range(len(BN))); a.set_xticklabels(BX, fontsize=8)
    a.set_ylabel("Recall@300 (IoU 0.5)"); a.set_xlabel("density bucket")
    a.set_ylim(0.5, 1.0); a.legend(fontsize=7.5)
    a.set_title("(a) recall collapses with density")
    sku = {"<50": (9.4, 4.6, 86.0), "50-100": (26.5, 8.5, 65.1),
           "100-150": (41.8, 12.0, 46.2), "150-300": (57.0, 12.2, 30.8),
           ">=300": (88.4, 2.6, 9.0)}
    tp = [sku[bk][0] for bk in BN]; dup = [sku[bk][1] for bk in BN]
    fp = [sku[bk][2] for bk in BN]; x = np.arange(len(BN))
    b.bar(x, tp, label="TP distinct (exhaustion)", color="#2c7fb8")
    b.bar(x, dup, bottom=tp, color="#7fcdbb", label="duplicate")
    b.bar(x, fp, bottom=np.array(tp) + np.array(dup), color="#edf8b1",
          label="FP (displacement)")
    b.set_xticks(x); b.set_xticklabels(BX, fontsize=8)
    b.set_ylabel("top-300 slot composition (%)"); b.set_xlabel("density bucket")
    b.set_title("(b) displacement to exhaustion"); b.legend(fontsize=7.5)
    b.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_teaser.pdf"))
    print("wrote fig_teaser.pdf")


if __name__ == "__main__":
    fig_teaser()
    fig_recall_density()
    fig_slot_composition()
    fig_attribution()
    fig_ap_budget()
    fig_mask()
    print("all figures written to", FIG)
