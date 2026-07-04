"""Per-GT local-crowding analysis: observational H1' vs H2 discriminator.

For every GT box, computes:
  - local crowding: max IoU with any other GT (occlusion proxy), and
    neighbor count = #GT centers within 2*sqrt(area) of its center
  - global context: image GT count (and its bucket)
  - outcome: matched within top-300 / top-1000 ranked predictions (IoU 0.5)

Outputs per-GT CSV + a stratified table recall@300 over
(local-crowding tertile x global-count bucket). If recall varies strongly
along the LOCAL axis conditional on global count -> supports H2 (assignment/
crowding failure). If it varies along the GLOBAL axis conditional on local
crowding -> supports H1' (global rank/budget pressure).

Usage: python wp1_local_density.py --dataset visdrone --gt <dir|odgt>
       --preds <jsonl> --out-prefix <prefix>
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

import wp1_eval as we


def per_gt_records(gt_data, preds_path, iou_thr=0.5):
    recs = []
    with open(preds_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            img = rec["image"]
            if img not in gt_data:
                continue
            g = gt_data[img]
            gt = g["gt"]
            n_gt = len(gt)
            if n_gt == 0:
                continue
            arr = np.array(rec["boxes"], dtype=np.float32).reshape(-1, 6)
            boxes = arr[:, :4]
            # deploy mode: all predictions occupy rank slots (review fix)
            matched_rank, _ = we.greedy_match(boxes, gt, iou_thr)

            # local crowding metrics
            ious = we.iou_matrix(gt, gt)
            np.fill_diagonal(ious, 0)
            max_iou = ious.max(axis=1) if n_gt > 1 else np.zeros(n_gt)
            ctr = np.stack([(gt[:, 0] + gt[:, 2]) / 2,
                            (gt[:, 1] + gt[:, 3]) / 2], axis=1)
            area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
            rad = 2 * np.sqrt(np.maximum(area, 1))
            d2 = ((ctr[:, None, :] - ctr[None, :, :]) ** 2).sum(-1)
            nbr = (d2 <= (rad[:, None] ** 2)).sum(axis=1) - 1

            for i in range(n_gt):
                recs.append({
                    "image": img, "n_gt": n_gt,
                    "bucket": we.bucket_of(n_gt),
                    "area": float(area[i]),
                    "size": ("small" if area[i] < 32**2 else
                             "medium" if area[i] < 96**2 else "large"),
                    "max_iou_nbr": round(float(max_iou[i]), 4),
                    "nbr_count": int(nbr[i]),
                    "rank": int(matched_rank[i]),
                    "m300": int(0 <= matched_rank[i] < 300),
                    "m1000": int(matched_rank[i] >= 0),
                })
    return recs


def stratified_table(recs, local_key, local_edges, label):
    print(f"\n=== recall@300 by {label} (rows) x global count (cols) ===")
    gbuckets = we.BUCKET_NAMES
    rowlab = "local x global"
    header = f"{rowlab:>16} | " + " | ".join(f"{b:>12}" for b in gbuckets)
    print(header)
    for lo, hi, lname in local_edges:
        cells = []
        for b in gbuckets:
            sel = [r for r in recs
                   if r["bucket"] == b and lo <= r[local_key] < hi]
            if len(sel) < 30:
                cells.append(f"{'--':>12}")
            else:
                r300 = np.mean([r["m300"] for r in sel])
                r1000 = np.mean([r["m1000"] for r in sel])
                cells.append(f"{r300:.3f}/+{(r1000 - r300):.3f}"[:12].rjust(12))
        print(f"{lname:>16} | " + " | ".join(cells))
    print("(cell = recall@300 / +gain from budget 1000; '--' if n<30)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["visdrone", "crowdhuman"],
                    required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    gt_data = (we.load_visdrone_gt(args.gt) if args.dataset == "visdrone"
               else we.load_crowdhuman_gt(args.gt))
    recs = per_gt_records(gt_data, args.preds, args.iou)
    print(f"{len(recs)} GT records")

    out_csv = args.out_prefix + "_per_gt.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    print("wrote", out_csv)

    stratified_table(recs, "max_iou_nbr",
                     [(0.0, 0.1, "isolated <0.1"),
                      (0.1, 0.3, "touch 0.1-0.3"),
                      (0.3, 0.6, "overlap .3-.6"),
                      (0.6, 1.01, "heavy >0.6")],
                     "GT-GT max IoU")
    stratified_table(recs, "nbr_count",
                     [(0, 1, "0 nbrs"), (1, 4, "1-3 nbrs"),
                      (4, 10, "4-9 nbrs"), (10, 10**9, ">=10 nbrs")],
                     "neighbor count")

    # small objects only (dominant class in dense aerial buckets)
    small = [r for r in recs if r["size"] == "small"]
    print(f"\n--- SMALL objects only ({len(small)} GT) ---")
    stratified_table(small, "nbr_count",
                     [(0, 1, "0 nbrs"), (1, 4, "1-3 nbrs"),
                      (4, 10, "4-9 nbrs"), (10, 10**9, ">=10 nbrs")],
                     "neighbor count (small only)")


if __name__ == "__main__":
    main()
