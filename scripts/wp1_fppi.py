"""FPPI-conditioned recall per density bucket: deployment-realistic check.

Counters the critique that budget relaxation 300->1000 merely admits garbage:
for each density bucket, sweep rank cutoff k and report recall at the k where
pooled false positives per image (FPPI) crosses {10, 50, 100}, plus recall and
FPPI at k=300 and k=1000 for reference.

A prediction is a TP if it greedy-matches an unmatched GT at IoU>=thr (ranked
by confidence); else FP (ignore-region preds dropped beforehand).

Usage: python wp1_fppi.py --dataset visdrone --gt <dir> --preds <jsonl>
"""
import argparse
import json

import numpy as np

import wp1_eval as we

KMAX = 1000
FPPI_LEVELS = [10.0, 50.0, 100.0]


def per_image_tp_flags(gt_data, preds_path, iou_thr=0.5):
    """Returns list of (bucket, n_gt, tp_flags, fp_eligible, n_det) per image.

    Deploy-faithful protocol (review fix): ALL predictions occupy rank slots;
    predictions whose center falls in an ignore region are exempt from FP
    counting (neither TP nor FP) but still consume their slot.
    """
    out = []
    with open(preds_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            img = rec["image"]
            if img not in gt_data:
                continue
            g = gt_data[img]
            gt = g["gt"]
            arr = np.array(rec["boxes"], dtype=np.float32).reshape(-1, 6)
            boxes = arr[:, :4]
            matched_rank, _ = we.greedy_match(boxes, gt, iou_thr)
            tp = np.zeros(KMAX, dtype=bool)
            for r in matched_rank:
                if 0 <= r < KMAX:
                    tp[r] = True
            n_det = min(len(boxes), KMAX)
            in_ignore = np.zeros(KMAX, dtype=bool)
            if len(g["ignore"]) > 0 and len(boxes) > 0:
                ctr = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2,
                                (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
                ii = we.centers_in_boxes(ctr, g["ignore"])
                in_ignore[:min(len(ii), KMAX)] = ii[:KMAX]
            fp_eligible = ~tp & ~in_ignore
            out.append((we.bucket_of(len(gt)), len(gt), tp, fp_eligible,
                        n_det))
    return out


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
    data = per_image_tp_flags(gt_data, args.preds, args.iou)

    print(f"\n=== rank-pooled FP/img-conditioned recall (IoU {args.iou}) ===")
    print("NOTE: rank-cutoff pooling, NOT literature FPPI/MR-2 "
          "(threshold-swept); ignore-region preds exempt from FP count.")
    hdr = (f"{'bucket':>9} {'n_img':>6} | " +
           " ".join(f"R@FP{int(l):<4}" for l in FPPI_LEVELS) +
           " |  R@300(FP/img)  R@1000(FP/img)")
    print(hdr)
    for bname in we.BUCKET_NAMES:
        rows = [d for d in data if d[0] == bname]
        if not rows:
            continue
        n_img = len(rows)
        tot_gt = sum(r[1] for r in rows)
        # cumulative pooled TP and FP as k sweeps 1..KMAX
        tp_mat = np.stack([r[2] for r in rows])          # (n_img, KMAX)
        fpel = np.stack([r[3] for r in rows])            # (n_img, KMAX)
        ndet = np.array([r[4] for r in rows])            # (n_img,)
        # det exists at rank r only if r < ndet
        exists = np.arange(KMAX)[None, :] < ndet[:, None]
        fp_mat = exists & fpel
        cum_tp = tp_mat.sum(0).cumsum()                  # pooled TP(k)
        cum_fp = fp_mat.sum(0).cumsum()                  # pooled FP(k)
        recall_k = cum_tp / max(tot_gt, 1)
        fppi_k = cum_fp / n_img
        cells = []
        for lvl in FPPI_LEVELS:
            idx = np.searchsorted(fppi_k, lvl, side="right") - 1
            cells.append(f"{recall_k[idx]:.3f}      " if idx >= 0
                         else "--         ")
        r300, f300 = recall_k[299], fppi_k[299]
        r1k, f1k = recall_k[-1], fppi_k[-1]
        print(f"{bname:>9} {n_img:>6} | " + " ".join(cells) +
              f" |  {r300:.3f}({f300:.0f})    {r1k:.3f}({f1k:.0f})")


if __name__ == "__main__":
    main()
