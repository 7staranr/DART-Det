# -*- coding: utf-8 -*-
"""Gate-realized AP for DABA.

Reported AP gains were measured under a constant relaxed cap (K=1000). This
driver measures AP under the *actual* per-image DABA budget K*(x), so the
headline AP gain can be attributed to the operator that the paper proposes
rather than to a constant cap. Reuses the verified matching/AP code path in
wp4_ap_bootstrap.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wp1_eval as we                      # noqa: E402
import wp4_ap_bootstrap as ab              # noqa: E402

TAU_P = 0.1
T1, T2 = 100, 200


def tier_of(count):
    """DABA rule, identical to scripts/wp4_budget_policy.py."""
    if count < T1:
        return 300
    if count < T2:
        return 600
    return 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["visdrone", "sku", "dota"], required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--thr-density", type=int, default=150)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    n_classes = 1 if args.dataset == "sku" else (15 if args.dataset == "dota" else 10)
    gt_data = (we.load_sku_gt(args.gt) if args.dataset == "sku"
               else we.load_yolo_gt(args.gt) if args.dataset == "dota"
               else we.load_visdrone_gt(args.gt))
    preds = ab.load_preds(args.preds)

    dense = [k for k, v in gt_data.items()
             if len(v["gt"]) >= args.thr_density and k in preds]
    print(f"dense images (GT>={args.thr_density}, with preds): {len(dense)}")

    # per-image DABA budget from the density proxy
    kstar, tiers = {}, {}
    for img in dense:
        n = int((preds[img][:, 4] >= TAU_P).sum())
        kstar[img] = tier_of(n)
        tiers[kstar[img]] = tiers.get(kstar[img], 0) + 1
    print("DABA budget distribution on the dense subset: "
          + ", ".join(f"K*={k}: {v} imgs" for k, v in sorted(tiers.items()))
          + f"   mean K* = {np.mean([kstar[i] for i in dense]):.1f}")

    ngt_cls = [{c: int((gt_data[i]["gt_cls"] == c).sum())
                for c in range(n_classes)
                if int((gt_data[i]["gt_cls"] == c).sum()) > 0} for i in dense]

    def cache_for(kfun):
        return [ab.match_image(preds[i], gt_data[i]["gt"], gt_data[i]["gt_cls"],
                               kfun(i), args.iou, n_classes) for i in dense]

    caches = {
        "K=300 (default)":  cache_for(lambda i: 300),
        "K=1000 (constant)": cache_for(lambda i: 1000),
        "DABA gate K*(x)":  cache_for(lambda i: kstar[i]),
    }

    full = np.arange(len(dense))
    pts = {k: ab.ap_from_cache(full, c, ngt_cls, n_classes) for k, c in caches.items()}
    print("\npoint AP@%.2f:" % args.iou)
    for k, v in pts.items():
        print(f"   {k:20s} {v:.4f}")
    base = pts["K=300 (default)"]
    print(f"   delta(gate - default)     {pts['DABA gate K*(x)'] - base:+.4f}")
    print(f"   delta(constant - default) {pts['K=1000 (constant)'] - base:+.4f}")

    rng = np.random.default_rng(args.seed)
    dg, dc = [], []
    for _ in range(args.nboot):
        s = rng.integers(0, len(dense), len(dense))
        b = ab.ap_from_cache(s, caches["K=300 (default)"], ngt_cls, n_classes)
        dg.append(ab.ap_from_cache(s, caches["DABA gate K*(x)"], ngt_cls, n_classes) - b)
        dc.append(ab.ap_from_cache(s, caches["K=1000 (constant)"], ngt_cls, n_classes) - b)
    for tag, d in (("DABA gate vs K=300", dg), ("constant K=1000 vs K=300", dc)):
        d = np.array(d)
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"\nbootstrap ({args.nboot}x images) {tag}: "
              f"mean {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"P(delta>0)={np.mean(d > 0):.3f}")


if __name__ == "__main__":
    main()
