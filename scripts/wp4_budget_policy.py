"""Density-adaptive Budget Allocation (DABA): a zero-training
inference policy, evaluated from cached ranked predictions.

Policies (per image, choose budget k in tiers {300, 600, 1200->1000}):
  fixed300   : k = 300 (YOLO26 default)
  fixed1000  : k = 1000 (naive expansion; FP cost shown for contrast)
  oracle     : k by true GT count   (<100 -> 300, <200 -> 600, else 1000)
  selfest    : k by ndet@conf0.10   (<100 -> 300, <200 -> 600, else 1000)
               -- deployable: the count of confident dets is available from
                  the head output itself at zero extra cost.

Metrics per density bucket: recall, FP/img (ignore-exempt), mean slots used.
Uses wp1_fppi.per_image_tp_flags (deploy-faithful TP/FP rank flags).

Usage: python wp4_budget_policy.py --dataset visdrone --gt <dir>
       --preds <jsonl> [--iou 0.5]
"""
import argparse
import json

import numpy as np

import wp1_eval as we
import wp1_fppi as wf

TIERS = [300, 600, 1000]


def tier_of(count):
    if count < 100:
        return 300
    if count < 200:
        return 600
    return 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["visdrone", "crowdhuman", "sku"],
                    required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    gt_data = (we.load_visdrone_gt(args.gt) if args.dataset == "visdrone"
               else we.load_sku_gt(args.gt) if args.dataset == "sku"
               else we.load_crowdhuman_gt(args.gt))
    data = wf.per_image_tp_flags(gt_data, args.preds, args.iou)

    # need ndet@conf0.1 per image for the self-estimating policy
    conf01 = {}
    with open(args.preds, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            arr = np.array(rec["boxes"], dtype=np.float32).reshape(-1, 6)
            conf01[rec["image"]] = int((arr[:, 4] >= 0.10).sum())
    # re-read image ids in same order as data
    ids = []
    with open(args.preds, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["image"] in gt_data:
                ids.append(rec["image"])

    policies = {
        "fixed300": lambda n_gt, nd01: 300,
        "fixed1000": lambda n_gt, nd01: 1000,
        "oracle": lambda n_gt, nd01: tier_of(n_gt),
        "selfest": lambda n_gt, nd01: tier_of(nd01),
    }

    print(f"\n=== density-adaptive budget policies (IoU {args.iou}) ===")
    print(f"{'policy':>10} {'bucket':>9} {'recall':>7} {'FP/img':>7} "
          f"{'slots':>6}")
    summary = {}
    for pname, pfun in policies.items():
        by_bucket = {}
        for (bucket, n_gt, tp, fpel, n_det), img_id in zip(data, ids):
            k = pfun(n_gt, conf01.get(img_id, 0))
            k_eff = min(k, n_det)
            rec_tp = int(tp[:k_eff].sum())
            rec_fp = int(fpel[:k_eff].sum())
            b = by_bucket.setdefault(bucket, [0, 0, 0, 0])  # tp, gt, fp, slots
            b[0] += rec_tp
            b[1] += n_gt
            b[2] += rec_fp
            b[3] += k_eff
        srow = {}
        for bname in we.BUCKET_NAMES:
            if bname not in by_bucket:
                continue
            tp_, gt_, fp_, sl_ = by_bucket[bname]
            n_img = sum(1 for (b, *_), _ in zip(data, ids) if b == bname)
            srow[bname] = (tp_ / max(gt_, 1), fp_ / n_img, sl_ / n_img)
            print(f"{pname:>10} {bname:>9} {tp_ / max(gt_, 1):7.4f} "
                  f"{fp_ / n_img:7.1f} {sl_ / n_img:6.0f}")
        summary[pname] = srow
        print()

    # headline: dense-bucket gain at what FP / slot cost
    for b in ("150-300", ">=300"):
        if b in summary["fixed300"]:
            r0, f0, s0 = summary["fixed300"][b]
            for p in ("oracle", "selfest"):
                r1, f1, s1 = summary[p][b]
                print(f"[{b}] {p}: recall {r0:.3f}->{r1:.3f} "
                      f"(+{100 * (r1 - r0):.1f}pts), FP/img {f0:.0f}->{f1:.0f}, "
                      f"slots {s0:.0f}->{s1:.0f}")


if __name__ == "__main__":
    main()
