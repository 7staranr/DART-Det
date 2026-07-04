"""Conf-floor sensitivity of the budget-recovery claim.

the headline "+17.5pt recall recovery" counts matches at conf~0.001 (the
oracle/full ranked list). The deployable recovery at a real operating point is
smaller. For each density bucket and conf floor t in {0.001,0.1,0.25}: keep
only preds with conf>=t (still ranked), then recall@300 vs recall@1000;
report the recovery R@1000-R@300. Shows the gain is real but conf-dependent.

Usage: python wp4_conf_floor.py --dataset {visdrone,sku} --preds <jsonl> --gt <path>
"""
import argparse
import json
from collections import defaultdict

import numpy as np

import wp1_eval as we

BUCKETS = [(0, 50), (50, 100), (100, 150), (150, 300), (300, 10**9)]
BNAMES = ["<50", "50-100", "100-150", "150-300", ">=300"]
FLOORS = [0.001, 0.10, 0.25]


def bucket_of(n):
    for (lo, hi), nm in zip(BUCKETS, BNAMES):
        if lo <= n < hi:
            return nm
    return BNAMES[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["visdrone", "sku"], required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()
    class_aware = args.dataset == "visdrone"
    gt_data = (we.load_sku_gt(args.gt) if args.dataset == "sku"
               else we.load_visdrone_gt(args.gt))

    # acc[bucket][floor] = [matched@300, matched@1000, n_gt]
    acc = defaultdict(lambda: defaultdict(lambda: np.zeros(3)))
    with open(args.preds, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            g = gt_data.get(r["image"])
            if g is None:
                continue
            gt, gcls = g["gt"], g["gt_cls"]
            b = bucket_of(len(gt))
            arr = np.array(r["boxes"], dtype=np.float32).reshape(-1, 6)
            confs = arr[:, 4]
            for t in FLOORS:
                keep = confs >= t
                p = arr[keep]
                boxes, pcls = p[:, :4], p[:, 5].astype(int)
                matched_rank = np.full(len(gt), -1)
                if len(boxes) and len(gt):
                    ious = we.iou_matrix(boxes, gt)
                    if class_aware:
                        ious = np.where(pcls[:, None] == gcls[None, :], ious, 0.0)
                    taken = np.zeros(len(gt), bool)
                    for ri in range(len(boxes)):
                        row = np.where(~taken, ious[ri], -1.0)
                        j = int(row.argmax())
                        if row[j] >= args.iou:
                            taken[j] = True
                            matched_rank[j] = ri
                m300 = ((matched_rank >= 0) & (matched_rank < 300)).sum()
                m1000 = ((matched_rank >= 0) & (matched_rank < 1000)).sum()
                acc[b][t] += [m300, m1000, len(gt)]

    print(f"\n=== budget recovery R@1000-R@300 by conf floor ({args.dataset}, "
          f"IoU {args.iou}) ===")
    hdr = f"{'bucket':>9} " + " ".join(
        f"{'R@300/R@1000/Δ @'+str(t):>22}" for t in FLOORS)
    print(hdr)
    for b in BNAMES:
        if b not in acc:
            continue
        cells = []
        for t in FLOORS:
            m3, m1k, ng = acc[b][t]
            r3, r1k = m3 / max(ng, 1), m1k / max(ng, 1)
            cells.append(f"{r3:.3f}/{r1k:.3f}/{r1k-r3:+.3f}".rjust(22))
        print(f"{b:>9} " + " ".join(cells))
    print("\n(conf=0.001 = oracle/full-list recovery [inflated]; conf>=0.1 / "
          ">=0.25 = deployable recovery. Headline should cite the conf>=0.1 "
          "value, not the 0.001 oracle.)")


if __name__ == "__main__":
    main()
