"""Density-stratified recall and budget statistics.

Protocol:
  - DEPLOY mode (default): ALL ranked predictions consume budget slots
    (truncate to k first). Ignore regions only exempt predictions from FP
    counting (in wp1_fppi), never from ranking. This is deployment-faithful.
  - CLEAN mode (--ignore-mode clean): legacy behavior (pre-filter ignore-region
    preds before ranking) for sensitivity comparison only.
  - Metric name: R@k (single-IoU recall at rank budget k), NOT COCO AR.
  - CrowdHuman: --box-field vbox (visible, default) | fbox (full-body,
    sensitivity) — fbox vs COCO visible-box convention inflates the density
    decline (review §2.4).
  - Bootstrap CI suppressed when n_images < 20 (degenerate otherwise).
  - rel_rec = (R@1000 - R@300) / (1 - R@300): relative recovery of misses from
    the rank tail (review §2.1: absolute deltas conflate with headroom).
  - dup-suspicion audit: TP whose IoU with an already-matched GT exceeds the
    IoU with its own assigned GT (double-counting signature, review §1.7).
  - mean_cache_depth reported per bucket: R@1000 is really R@(cache depth)
    when depth < 1000 (review §1.5).
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

BUCKETS = [(0, 50), (50, 100), (100, 150), (150, 300), (300, 10**9)]
BUCKET_NAMES = ["<50", "50-100", "100-150", "150-300", ">=300"]
KS = [100, 300, 600, 1000]
SAT_CONFS = [0.25, 0.10]
CAP = 300
AREA_RANGES = {"small": (0, 32**2), "medium": (32**2, 96**2),
               "large": (96**2, 10**12)}
MIN_IMAGES_FOR_CI = 20


def bucket_of(n):
    for (lo, hi), name in zip(BUCKETS, BUCKET_NAMES):
        if lo <= n < hi:
            return name
    return BUCKET_NAMES[-1]


def load_visdrone_gt(ann_dir):
    """{image_id: {'gt': Nx4 xyxy, 'gt_cls': N (0-9), 'ignore': Mx4}}.
    GT score field is uniformly 1 for cat 1-10 in val (verified 2026-06-11),
    so no score-based ignore handling is required. gt_cls = category-1,
    matching the YOLO training label convention."""
    out = {}
    for fn in os.listdir(ann_dir):
        if not fn.endswith(".txt"):
            continue
        gt, cls, ign = [], [], []
        with open(os.path.join(ann_dir, fn), "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip().split(",")
                if len(p) < 6:
                    continue
                x, y, w, h, cat = (float(p[0]), float(p[1]), float(p[2]),
                                   float(p[3]), int(p[5]))
                box = [x, y, x + w, y + h]
                if 1 <= cat <= 10:
                    gt.append(box)
                    cls.append(cat - 1)
                elif cat in (0, 11):
                    ign.append(box)
        out[fn[:-4]] = {"gt": np.array(gt, dtype=np.float32).reshape(-1, 4),
                        "gt_cls": np.array(cls, dtype=np.int32),
                        "ignore": np.array(ign, dtype=np.float32).reshape(-1, 4)}
    return out


def load_sku_gt(csv_path):
    """SKU-110K CSV: image_name,x1,y1,x2,y2,class,w,h. Single class, no
    ignore regions. Keys are image stems (e.g. val_0)."""
    import csv as _csv
    from collections import defaultdict
    boxes = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in _csv.reader(f):
            if len(row) < 8:
                continue
            name, x1, y1, x2, y2 = row[0], *map(float, row[1:5])
            if x2 > x1 and y2 > y1:
                boxes[os.path.splitext(name)[0]].append([x1, y1, x2, y2])
    return {k: {"gt": np.array(v, dtype=np.float32),
                "gt_cls": np.zeros(len(v), dtype=np.int32),
                "ignore": np.zeros((0, 4), dtype=np.float32)}
            for k, v in boxes.items()}


def load_yolo_gt(label_dir, imgsz=1024):
    """Generic YOLO-format labels (cls xc yc w h, normalized) -> xyxy in pixels.
    Used for DOTAv1-tiled (fixed 1024x1024 crops). No ignore regions."""
    out = {}
    for fn in os.listdir(label_dir):
        if not fn.endswith(".txt"):
            continue
        gt, cls = [], []
        with open(os.path.join(label_dir, fn), "r", encoding="utf-8") as f:
            for line in f:
                p = line.split()
                if len(p) < 5:
                    continue
                c, xc, yc, w, h = int(p[0]), *map(float, p[1:5])
                x1 = (xc - w / 2) * imgsz
                y1 = (yc - h / 2) * imgsz
                x2 = (xc + w / 2) * imgsz
                y2 = (yc + h / 2) * imgsz
                if x2 > x1 and y2 > y1:
                    gt.append([x1, y1, x2, y2])
                    cls.append(c)
        out[fn[:-4]] = {"gt": np.array(gt, dtype=np.float32).reshape(-1, 4),
                        "gt_cls": np.array(cls, dtype=np.int32),
                        "ignore": np.zeros((0, 4), dtype=np.float32)}
    return out


def load_crowdhuman_gt(odgt_path, box_field="vbox"):
    """box_field: vbox (visible region, matches COCO convention) or fbox
    (amodal full-body). Ignored entries (tag!=person or extra.ignore) go to
    the ignore list using the same field."""
    out = {}
    with open(odgt_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            gt, ign = [], []
            for gb in rec.get("gtboxes", []):
                bf = gb.get(box_field) or gb.get("fbox")
                x, y, w, h = bf
                box = [x, y, x + w, y + h]
                if gb.get("tag") == "person" and \
                   gb.get("extra", {}).get("ignore", 0) != 1:
                    gt.append(box)
                else:
                    ign.append(box)
            out[rec["ID"]] = {
                "gt": np.array(gt, dtype=np.float32).reshape(-1, 4),
                "ignore": np.array(ign, dtype=np.float32).reshape(-1, 4)}
    return out


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0).astype(np.float32)


def centers_in_boxes(pts, boxes):
    if len(pts) == 0 or len(boxes) == 0:
        return np.zeros(len(pts), dtype=bool)
    x, y = pts[:, 0:1], pts[:, 1:2]
    inside = ((x >= boxes[:, 0]) & (x <= boxes[:, 2]) &
              (y >= boxes[:, 1]) & (y <= boxes[:, 3]))
    return inside.any(axis=1)


def greedy_match(pred_boxes, gt_boxes, iou_thr, pred_cls=None, gt_cls=None):
    """Preds sorted by conf desc. Returns (matched_rank, dup_suspect):
    matched_rank[g] = rank of pred matched to GT g else -1;
    dup_suspect[g] = True if at match time the pred's IoU with some
    ALREADY-MATCHED GT exceeded its IoU with g (double-count signature).
    If pred_cls/gt_cls given, a pred may only match a same-class GT."""
    matched_rank = np.full(len(gt_boxes), -1, dtype=np.int32)
    dup_suspect = np.zeros(len(gt_boxes), dtype=bool)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return matched_rank, dup_suspect
    ious = iou_matrix(pred_boxes, gt_boxes)
    if pred_cls is not None and gt_cls is not None:
        ious = np.where(pred_cls[:, None] == gt_cls[None, :], ious, 0.0)
    gt_taken = np.zeros(len(gt_boxes), dtype=bool)
    for r in range(len(pred_boxes)):
        row = ious[r]
        avail = np.where(~gt_taken, row, -1.0)
        g = int(avail.argmax())
        if avail[g] >= iou_thr:
            if gt_taken.any():
                taken_max = row[gt_taken].max()
                if taken_max > row[g]:
                    dup_suspect[g] = True
            gt_taken[g] = True
            matched_rank[g] = r
    return matched_rank, dup_suspect


def evaluate(gt_data, preds_path, iou_thr=0.5, ignore_mode="deploy",
             class_aware=False):
    per_image = []
    with open(preds_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            img = rec["image"]
            if img not in gt_data:
                continue
            g = gt_data[img]
            gt = g["gt"]
            arr = np.array(rec["boxes"], dtype=np.float32).reshape(-1, 6)
            boxes, confs = arr[:, :4], arr[:, 4]
            pcls = arr[:, 5].astype(np.int32)
            if ignore_mode == "clean" and len(g["ignore"]) > 0 and len(boxes):
                ctr = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2,
                                (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
                keep = ~centers_in_boxes(ctr, g["ignore"])
                boxes, confs, pcls = boxes[keep], confs[keep], pcls[keep]
            # deploy mode: nothing filtered — every prediction occupies a slot

            if class_aware:
                matched_rank, dup_suspect = greedy_match(
                    boxes, gt, iou_thr, pred_cls=pcls,
                    gt_cls=g.get("gt_cls"))
            else:
                matched_rank, dup_suspect = greedy_match(boxes, gt, iou_thr)
            n_gt = len(gt)
            areas = ((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
                     if n_gt else np.zeros(0))

            row = {"image": img, "n_gt": n_gt,
                   "bucket": bucket_of(n_gt),
                   "cache_depth": int(len(boxes))}
            for k in KS:
                row[f"matched@{k}"] = int(((matched_rank >= 0) &
                                           (matched_rank < k)).sum())
            tail_tp = (matched_rank >= 300) & (matched_rank < 1000)
            row["tail_tp"] = int(tail_tp.sum())
            row["tail_tp_dupsus"] = int((tail_tp & dup_suspect).sum())
            for t in SAT_CONFS:
                nd = int((confs >= t).sum())
                row[f"ndet@conf{t}"] = nd
                row[f"sat@conf{t}"] = int(nd >= CAP)
            for sz, (lo, hi) in AREA_RANGES.items():
                m = (areas >= lo) & (areas < hi)
                row[f"n_gt_{sz}"] = int(m.sum())
                row[f"matched300_{sz}"] = int(((matched_rank >= 0) &
                                               (matched_rank < 300) & m).sum())
            per_image.append(row)
    return per_image


def bootstrap_ci(rows, key, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.array([r[key] for r in rows], dtype=np.float64)
    g = np.array([r["n_gt"] for r in rows], dtype=np.float64)
    n = len(rows)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        gs = g[idx].sum()
        stats.append(m[idx].sum() / gs if gs > 0 else np.nan)
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def aggregate(per_image):
    by_bucket = defaultdict(list)
    for r in per_image:
        by_bucket[r["bucket"]].append(r)
    table = []
    for bname in BUCKET_NAMES:
        rows = by_bucket.get(bname, [])
        if not rows:
            continue
        n_img = len(rows)
        tot_gt = sum(r["n_gt"] for r in rows)
        agg = {"bucket": bname, "n_images": n_img, "total_gt": tot_gt,
               "mean_cache_depth": round(
                   float(np.mean([r["cache_depth"] for r in rows])), 1)}
        for k in KS:
            agg[f"R@{k}"] = round(
                sum(r[f"matched@{k}"] for r in rows) / max(tot_gt, 1), 4)
        r300, r1000 = agg["R@300"], agg["R@1000"]
        agg["rel_rec"] = (round((r1000 - r300) / (1 - r300), 4)
                          if r300 < 1 else None)
        if n_img >= MIN_IMAGES_FOR_CI:
            lo, hi = bootstrap_ci(rows, "matched@300")
            agg["R@300_ci_lo"], agg["R@300_ci_hi"] = lo, hi
        else:
            agg["R@300_ci_lo"] = agg["R@300_ci_hi"] = None
        tail = sum(r["tail_tp"] for r in rows)
        agg["tail_tp"] = tail
        agg["tail_dupsus_frac"] = (round(
            sum(r["tail_tp_dupsus"] for r in rows) / tail, 4) if tail else None)
        for t in SAT_CONFS:
            agg[f"sat_rate@conf{t}"] = round(
                sum(r[f"sat@conf{t}"] for r in rows) / n_img, 4)
            agg[f"mean_ndet@conf{t}"] = round(
                float(np.mean([r[f"ndet@conf{t}"] for r in rows])), 1)
        for sz in AREA_RANGES:
            tg = sum(r[f"n_gt_{sz}"] for r in rows)
            ms = sum(r[f"matched300_{sz}"] for r in rows)
            agg[f"R300_{sz}"] = round(ms / tg, 4) if tg else None
            agg[f"n_gt_{sz}"] = tg
        table.append(agg)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["visdrone", "crowdhuman", "sku", "dota"],
                    required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--ignore-mode", choices=["deploy", "clean"],
                    default="deploy")
    ap.add_argument("--box-field", choices=["vbox", "fbox"], default="vbox",
                    help="crowdhuman GT box convention (vbox=visible)")
    ap.add_argument("--class-aware", action="store_true",
                    help="require pred class == GT class (visdrone FT only)")
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    gt_data = (load_visdrone_gt(args.gt) if args.dataset == "visdrone"
               else load_sku_gt(args.gt) if args.dataset == "sku"
               else load_yolo_gt(args.gt) if args.dataset == "dota"
               else load_crowdhuman_gt(args.gt, args.box_field))
    print(f"GT loaded: {len(gt_data)} images "
          f"(mode={args.ignore_mode}"
          + (f", box={args.box_field}" if args.dataset == "crowdhuman" else "")
          + ")")

    per_image = evaluate(gt_data, args.preds, iou_thr=args.iou,
                         ignore_mode=args.ignore_mode,
                         class_aware=args.class_aware)
    print(f"evaluated: {len(per_image)} images")

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    pi_path = args.out_prefix + "_per_image.csv"
    with open(pi_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
        w.writeheader()
        w.writerows(per_image)

    table = aggregate(per_image)
    tb_path = args.out_prefix + "_buckets.csv"
    with open(tb_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)

    print(f"\n=== density-stratified recall (IoU {args.iou}, "
          f"{args.ignore_mode} mode) ===")
    cols = (["bucket", "n_images", "total_gt", "mean_cache_depth"] +
            [f"R@{k}" for k in KS] +
            ["rel_rec", "tail_dupsus_frac", "sat_rate@conf0.25"])
    print("  ".join(f"{c:>16}" for c in cols))
    for row in table:
        print("  ".join(f"{str(row[c]):>16}" for c in cols))
    print(f"\nwrote {pi_path}\nwrote {tb_path}")


if __name__ == "__main__":
    main()
