"""FPPI-conditioned recall per density bucket: deployment-realistic check.

Counters the critique that budget relaxation 300->1000 merely admits garbage:
for each density bucket, sweep rank cutoff k and report recall at the k where
pooled false positives per image (FPPI) crosses {10, 50, 100}, plus recall and
FPPI at k=300 and k=1000 for reference.

A prediction is a TP if it greedy-matches an unmatched GT at IoU>=thr (ranked
by confidence). Predictions falling in an ignore region still consume their
rank slot but are exempt from the FP count -- they are not dropped beforehand,
which would understate how full the budget is. Everything else counts as FP.

Usage: python wp1_fppi.py --dataset visdrone --gt <dir> --preds <jsonl>
"""
import argparse
import json

import numpy as np

import wp1_eval as we

KMAX = 1000
FPPI_LEVELS = [10.0, 50.0, 100.0]


def per_image_tp_flags(gt_data, preds_path, iou_thr=0.5, with_ids=False):
    """Returns list of (bucket, n_gt, tp_flags, fp_eligible, n_det) per image.

    With with_ids=True each element is (image_id, row) instead. Callers that
    need the id must use this rather than rebuilding a parallel list of ids in
    a second pass: aligning two independently constructed lists by position is
    how a missing image got silently paired with another image's statistics.

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
            row = (we.bucket_of(len(gt)), len(gt), tp, fp_eligible, n_det)
            out.append((img, row) if with_ids else row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["visdrone", "crowdhuman", "sku"],
                    required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--allow-missing", action="store_true",
                    help="score GT images with no prediction record as zero "
                         "detections instead of failing")
    args = ap.parse_args()

    gt_data = (we.load_visdrone_gt(args.gt) if args.dataset == "visdrone"
               else we.load_sku_gt(args.gt) if args.dataset == "sku"
               else we.load_crowdhuman_gt(args.gt))

    # Same integrity guards as every other entry point: a GT image absent from
    # the cache would leave the FPPI denominator instead of scoring zero, and a
    # duplicated key would be counted twice. Neither is visible in the output.
    with open(args.preds, "r", encoding="utf-8") as f:
        _keys = [json.loads(line)["image"] for line in f if line.strip()]
    we.reject_duplicate_keys(_keys, args.preds)
    _missing = we.require_coverage(gt_data, _keys, args.allow_missing,
                                   what="FPPI-conditioned recall")

    data = per_image_tp_flags(gt_data, args.preds, args.iou)
    for _img in _missing:
        _g = gt_data[_img]
        data.append((we.bucket_of(len(_g["gt"])), len(_g["gt"]),
                     np.zeros(KMAX, dtype=bool), np.zeros(KMAX, dtype=bool), 0))

    print(f"\n=== rank-pooled FP/img-conditioned recall (IoU {args.iou}) ===")
    print("NOTE: rank-cutoff pooling, NOT literature FPPI/MR-2 "
          "(threshold-swept); ignore-region preds exempt from FP count.")
    print("MATCHING: class-agnostic (localization-only). The slot-composition, "
          "confidence-floor and AP scripts match class-aware on VisDrone, so "
          "recall here is not directly comparable with theirs.")
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
