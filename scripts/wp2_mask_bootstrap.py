"""Image-clustered bootstrap of the mask intervention, miss-subgroup.

The headline cohort-mean gap (+0.0706) mixes hit-subgroup (where masking
*removes* helpful context, negative dconf) and miss-subgroup (the recall-
relevant one). The honest causal quantity for recall is: among probes MISSED
at top-300, does removing competitors lift the target's score MORE in dense
images than the placebo (sparse) masking? Recompute restricted to miss probes,
with bootstrap over IMAGES (the cluster), not probes. Also the deploy-relevant
"newly detectable" rate (masked_conf crosses a threshold) at several thresholds.

Usage: python wp2_mask_bootstrap.py --csv <mask_formal.csv> [--nboot 5000]
"""
import argparse
import csv
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--nboot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    # group by image within cohort
    cells = {"dense": defaultdict(list), "sparse_placebo": defaultdict(list)}
    for r in rows:
        coh = r["cohort"]
        if coh not in cells:
            continue
        if int(r["hit300_full"]) == 1:        # miss-subgroup only
            continue
        cells[coh][r["image"]].append({
            "dconf": float(r["dconf"]),
            "full_conf": float(r["full_conf"]),
            "masked_conf": float(r["masked_conf"]),
        })

    def cohort_arrays(coh):
        imgs = list(cells[coh])
        return imgs, cells[coh]

    di, dmap = cohort_arrays("dense")
    pi, pmap = cohort_arrays("sparse_placebo")
    n_dense_probes = sum(len(v) for v in dmap.values())
    n_plac_probes = sum(len(v) for v in pmap.values())
    print(f"miss-subgroup probes: dense={n_dense_probes} "
          f"({len(di)} imgs), placebo={n_plac_probes} ({len(pi)} imgs)")

    def mean_dconf(imgs, mp, idx):
        vals = [d["dconf"] for i in idx for d in mp[imgs[i]]]
        return np.mean(vals) if vals else np.nan

    def newly(imgs, mp, idx, thr):
        tot = [d for i in idx for d in mp[imgs[i]]]
        if not tot:
            return np.nan
        return np.mean([d["masked_conf"] >= thr for d in tot])

    rng = np.random.default_rng(args.seed)
    obs_gap = mean_dconf(di, dmap, range(len(di))) - \
        mean_dconf(pi, pmap, range(len(pi)))
    gaps = []
    for _ in range(args.nboot):
        ds = rng.integers(0, len(di), len(di))
        ps = rng.integers(0, len(pi), len(pi))
        gaps.append(mean_dconf(di, dmap, ds) - mean_dconf(pi, pmap, ps))
    gaps = np.array(gaps)
    lo, hi = np.nanpercentile(gaps, [2.5, 97.5])
    print(f"\nmiss-subgroup dconf gap (dense - placebo), image-clustered "
          f"bootstrap {args.nboot}x:")
    print(f"  observed {obs_gap:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"P(gap>0)={np.mean(gaps > 0):.3f}")
    print(f"  dense miss mean dconf = "
          f"{mean_dconf(di, dmap, range(len(di))):+.4f}  "
          f"placebo = {mean_dconf(pi, pmap, range(len(pi))):+.4f}")

    print("\nnewly-detectable rate among missed probes (masked_conf>=thr):")
    print(f"{'thr':>6} {'dense':>8} {'placebo':>8}")
    for thr in (0.05, 0.10, 0.25):
        # image-clustered CI on dense newly rate
        nd = newly(di, dmap, range(len(di)), thr)
        npl = newly(pi, pmap, range(len(pi)), thr)
        boot = []
        for _ in range(args.nboot):
            ds = rng.integers(0, len(di), len(di))
            boot.append(newly(di, dmap, ds, thr))
        blo, bhi = np.nanpercentile(boot, [2.5, 97.5])
        print(f"{thr:>6} {nd:>7.1%} [{blo:.1%},{bhi:.1%}] {npl:>7.1%}")
    print("\n(If the clustered miss-subgroup gap CI excludes 0, the causal "
          "competitor-suppression claim survives at the recall-relevant "
          "subgroup, at its honest [smaller] effect size.)")


if __name__ == "__main__":
    main()
