"""Image-block permutation McNemar + BH-FDR.

Naive McNemar (wp5_mcnemar.py) treats ~38k per-GT outcomes as independent
paired trials; GTs in the same image are correlated (design effect can be ~8x),
and 6 strata are tested with no multiplicity control. This recomputes the WP5
treat-vs-base detection comparison with the IMAGE as the exchangeable unit.

Test: per GT, discordant direction d = +1 (treat detects@300, base misses),
-1 (base detects, treat misses), 0 (concordant). Observed stratum statistic =
sum(d). Under H0 (no model effect) the two models are exchangeable WITHIN each
image, so we flip the sign of an entire image's d-vector with prob 0.5 and
rebuild the null distribution (10k permutations). Two-sided p = P(|null|>=|obs|).
BH-FDR across the 6 strata. Pools multiple seeds by treating (seed,image) as
the cluster.

Usage: python wp5_mcnemar_clustered.py --pairs base1.csv treat1.csv [base2 treat2 ...]
"""
import argparse
import csv
from collections import defaultdict

import numpy as np

STRATA = {
    "ALL GT": lambda r: True,
    "sparse <50": lambda r: int(r["n_gt"]) < 50,
    "dense >=150": lambda r: int(r["n_gt"]) >= 150,
    "crowded >=0.3": lambda r: float(r["max_iou_nbr"]) >= 0.3,
    "dense AND crowded": lambda r: int(r["n_gt"]) >= 150
    and float(r["max_iou_nbr"]) >= 0.3,
    "small AND dense": lambda r: r["size"] == "small"
    and int(r["n_gt"]) >= 150,
}
SNAMES = list(STRATA)


def load(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build(pairs):
    """Returns {cluster_id: list of (d, strata_mask)} where cluster = (seed,img)."""
    clusters = defaultdict(list)
    for si, (bp, tp) in enumerate(pairs):
        base, treat = load(bp), load(tp)
        assert len(base) == len(treat)
        for rb, rt in zip(base, treat):
            assert rb["image"] == rt["image"]
            mb, mt = int(rb["m300"]), int(rt["m300"])
            if mb == mt:
                continue
            d = 1 if mt == 1 else -1
            mask = tuple(STRATA[s](rb) for s in SNAMES)
            clusters[(si, rb["image"])].append((d, mask))
    return clusters


def stratum_stats(clusters, signs=None):
    tot = np.zeros(len(SNAMES))
    for ci, (cid, items) in enumerate(clusters):
        s = signs[ci] if signs is not None else 1
        for d, mask in items:
            for j, inb in enumerate(mask):
                if inb:
                    tot[j] += s * d
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="base1 treat1 [base2 treat2 ...]")
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert len(args.pairs) % 2 == 0
    pairs = [(args.pairs[i], args.pairs[i + 1])
             for i in range(0, len(args.pairs), 2)]

    clusters = list(build(pairs).items())
    n_clusters = len(clusters)
    # also raw discordant counts for reporting
    raw = np.zeros((len(SNAMES), 2))  # treat_win, base_win
    for _, items in clusters:
        for d, mask in items:
            for j, inb in enumerate(mask):
                if inb:
                    raw[j, 0 if d > 0 else 1] += 1

    obs = stratum_stats(clusters)
    rng = np.random.default_rng(args.seed)
    null = np.zeros((args.n_perm, len(SNAMES)))
    for k in range(args.n_perm):
        signs = rng.choice([-1, 1], n_clusters)
        null[k] = stratum_stats(clusters, signs)
    pvals = np.array([
        (np.abs(null[:, j]) >= abs(obs[j])).mean() for j in range(len(SNAMES))])

    # BH-FDR
    order = np.argsort(pvals)
    m = len(pvals)
    bh = np.empty(m)
    prev = 1.0
    for rank, idx in enumerate(reversed(order)):
        i = m - rank
        prev = min(prev, pvals[idx] * m / i)
        bh[idx] = prev

    print(f"\n=== image-block permutation McNemar ({args.n_perm} perms, "
          f"{n_clusters} image-clusters, {len(pairs)} seed(s)) ===")
    print(f"{'stratum':>20} {'treat_win':>9} {'base_win':>8} {'net':>6} "
          f"{'perm_p':>8} {'BH_FDR':>8}  signif")
    for j, s in enumerate(SNAMES):
        sig = "*" if bh[j] < 0.05 else ""
        print(f"{s:>20} {int(raw[j,0]):>9} {int(raw[j,1]):>8} "
              f"{int(obs[j]):>+6} {pvals[j]:>8.3f} {bh[j]:>8.3f}  {sig}")
    print("\n(net = treat_win - base_win; clustering by image; H0 = models "
          "exchangeable within image. No stratum significant after FDR => the "
          "WP5 detection effect is not robust.)")


if __name__ == "__main__":
    main()
