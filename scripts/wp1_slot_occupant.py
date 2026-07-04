"""Top-300 budget-slot composition decomposition (distinct-TP vs duplicate vs FP).

In a dense image where the 300-cap binds, WHO occupies the 300 slots?
  - TP-distinct (matches a not-yet-matched GT)  -> real objects fill the budget = H1 (hard exhaustion)
  - duplicate    (matches an already-matched GT) -> wasted slot               = H1' (rank displacement)
  - FP           (matches no GT)                 -> garbage displaces a TP     = H1' (rank displacement)
Reports mean slot composition per density bucket, plus, among GTs MISSED at 300
but recovered by 1000, the composition of the 300 slots ranked above their
recovering prediction. This gives a measured H1/H1' split instead of "two
independent confirmations."

Usage: python wp1_slot_occupant.py --dataset {visdrone,sku} --preds <jsonl> --gt <path>
"""
import argparse
import json
from collections import defaultdict

import numpy as np

import wp1_eval as we

BUCKETS = [(0, 50), (50, 100), (100, 150), (150, 300), (300, 10**9)]
BNAMES = ["<50", "50-100", "100-150", "150-300", ">=300"]


def bucket_of(n):
    for (lo, hi), nm in zip(BUCKETS, BNAMES):
        if lo <= n < hi:
            return nm
    return BNAMES[-1]


def classify(preds, gt, gcls, k, iou_thr, class_aware):
    """Greedy-match top-k ranked preds; label each 0=FP,1=TP,2=dup. Also return
    matched_rank per GT (rank of the pred that first matches it)."""
    p = preds[:k]
    if len(p) == 0:
        return np.zeros(0, np.int8), np.full(len(gt), -1)
    boxes, pcls = p[:, :4], p[:, 5].astype(int)
    labels = np.zeros(len(boxes), np.int8)
    matched_rank = np.full(len(gt), -1)
    ious = we.iou_matrix(boxes, gt)
    if class_aware:
        ious = np.where(pcls[:, None] == gcls[None, :], ious, 0.0)
    taken = np.zeros(len(gt), bool)
    for r in range(len(boxes)):
        row = ious[r]
        if len(gt):
            avail = np.where(~taken, row, -1.0)
            j = int(avail.argmax())
            if avail[j] >= iou_thr:
                taken[j] = True
                matched_rank[j] = r
                labels[r] = 1
                continue
            if row.max() >= iou_thr:    # matches some (already-taken) GT
                labels[r] = 2
                continue
        labels[r] = 0
    return labels, matched_rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["visdrone", "sku"], required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--cap", type=int, default=300)
    args = ap.parse_args()
    class_aware = args.dataset == "visdrone"

    gt_data = (we.load_sku_gt(args.gt) if args.dataset == "sku"
               else we.load_visdrone_gt(args.gt))

    by_bucket = defaultdict(lambda: np.zeros(3))   # tp, dup, fp counts
    by_bucket_n = defaultdict(int)
    occ_above = defaultdict(lambda: np.zeros(3))    # for recovered-GTs
    occ_above_n = defaultdict(int)

    with open(args.preds, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            img = r["image"]
            g = gt_data.get(img)
            if g is None:
                continue
            preds = np.array(r["boxes"], dtype=np.float32).reshape(-1, 6)
            gt, gcls = g["gt"], g["gt_cls"]
            b = bucket_of(len(gt))
            # slot composition of the top-cap
            lab, mr_cap = classify(preds, gt, gcls, args.cap, args.iou,
                                   class_aware)
            if len(lab):
                comp = np.array([(lab == 1).sum(), (lab == 2).sum(),
                                 (lab == 0).sum()], float)
                by_bucket[b] += comp
                by_bucket_n[b] += 1
            # recovered GTs: matched in (cap, 1000]
            lab1k, mr1k = classify(preds, gt, gcls, 1000, args.iou, class_aware)
            recovered = np.where((mr1k >= args.cap) & (mr1k < 1000))[0]
            for gi in recovered:
                rr = mr1k[gi]
                above = lab1k[:args.cap]   # the cap slots all rank above rr>=cap
                occ_above[b] += np.array(
                    [(above == 1).sum(), (above == 2).sum(),
                     (above == 0).sum()], float)
                occ_above_n[b] += 1

    print(f"\n=== top-{args.cap} budget-slot composition by density (IoU "
          f"{args.iou}, {'class-aware' if class_aware else 'class-agnostic'}) ===")
    print(f"{'bucket':>9} {'n_img':>6} {'%TP-distinct':>12} {'%dup':>7} "
          f"{'%FP':>7}   interpretation")
    for b in BNAMES:
        if by_bucket_n[b] == 0:
            continue
        comp = by_bucket[b] / by_bucket[b].sum()
        h1 = comp[0]            # real distinct objects = hard exhaustion
        h1p = comp[1] + comp[2]  # dup+FP = displacement
        tag = "H1 (real objects fill cap)" if h1 > 0.6 else (
            "H1' (FP/dup displace)" if h1p > 0.5 else "mixed")
        print(f"{b:>9} {by_bucket_n[b]:>6} {100*comp[0]:>11.1f}% "
              f"{100*comp[1]:>6.1f}% {100*comp[2]:>6.1f}%   {tag}")

    print(f"\n=== among GTs recovered at 300<rank<=1000: composition of the "
          f"{args.cap} slots ranked above them ===")
    print(f"{'bucket':>9} {'n_GT':>6} {'%TP':>7} {'%dup':>7} {'%FP':>7}")
    for b in BNAMES:
        if occ_above_n[b] == 0:
            continue
        c = occ_above[b] / occ_above[b].sum()
        print(f"{b:>9} {occ_above_n[b]:>6} {100*c[0]:>6.1f}% {100*c[1]:>6.1f}% "
              f"{100*c[2]:>6.1f}%")
    print("\n(High %TP-distinct in the cap => real objects exhaust the budget "
          "(H1). High %FP+dup => recoverable TPs are displaced by garbage (H1').)")


if __name__ == "__main__":
    main()
