"""Causal density intervention via context masking.

For probe GT objects in dense images, gray-fill everything outside the
probe's local window (3x bbox, min 96px), keeping image size, scale and
letterbox identical. The probe's pixels and effective resolution are
unchanged; only competing objects are removed. Compare the probe's best
detection (IoU>=0.5) confidence and rank between full and masked runs.

Placebo control: same masking geometry applied to probes in SPARSE images
(<50 GT) measures the score drift caused by masking itself (context loss,
BN statistics), independent of competitor removal.

Interpretation:
  dense Δscore >> placebo Δscore  -> competitor displacement is causal (H1'/
                                     ranking mechanism real)
  dense Δscore ≈ placebo Δscore   -> dense objects are intrinsically hard;
                                     "rank displacement" deflates (H3-like)

Usage:
  python wp2_mask_intervention.py --model <pt> --dataset visdrone
      --gt <ann_dir> --images <img_dir> --out <csv> [--device 0]
      [--n-dense 30] [--probes-per-img 16]
"""
import argparse
import csv
import os

import numpy as np

import wp1_eval as we


def pick_probes(gt, matched_rank, n_probes, rng):
    """Sample up to n_probes GT indices, stratified: half missed@300, half
    hit@300 (rank<300)."""
    missed = np.where((matched_rank < 0) | (matched_rank >= 300))[0]
    hit = np.where((matched_rank >= 0) & (matched_rank < 300))[0]
    take_m = min(len(missed), n_probes // 2)
    take_h = min(len(hit), n_probes - take_m)
    sel = np.concatenate([
        rng.choice(missed, take_m, replace=False) if take_m else [],
        rng.choice(hit, take_h, replace=False) if take_h else [],
    ]).astype(int)
    return sel


def mask_image(img, box, scale=3.0, min_win=96, fill=114):
    """Gray-fill outside the probe window. box = xyxy."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = max(x2 - x1, 8), max(y2 - y1, 8)
    win_w, win_h = max(bw * scale, min_win), max(bh * scale, min_win)
    wx1 = int(max(0, cx - win_w / 2))
    wy1 = int(max(0, cy - win_h / 2))
    wx2 = int(min(w, cx + win_w / 2))
    wy2 = int(min(h, cy + win_h / 2))
    out = np.full_like(img, fill)
    out[wy1:wy2, wx1:wx2] = img[wy1:wy2, wx1:wx2]
    return out


def best_det_for_gt(boxes, confs, gt_box, iou_thr=0.5):
    """Highest-confidence prediction with IoU>=thr to gt_box.
    Returns (conf, rank) or (0.0, -1)."""
    if len(boxes) == 0:
        return 0.0, -1
    ious = we.iou_matrix(boxes, gt_box[None, :])[:, 0]
    ok = np.where(ious >= iou_thr)[0]
    if len(ok) == 0:
        return 0.0, -1
    r = int(ok[0])  # boxes sorted by conf desc -> first qualifying = best
    return float(confs[r]), r


def run(model, img_arr, imgsz, device, max_det=1000):
    res = model.predict(img_arr, imgsz=imgsz, conf=0.001, max_det=max_det,
                        device=device, verbose=False)[0]
    b = res.boxes
    xyxy = b.xyxy.cpu().numpy()
    conf = b.conf.cpu().numpy()
    order = conf.argsort()[::-1]
    return xyxy[order], conf[order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--n-dense", type=int, default=30)
    ap.add_argument("--n-sparse", type=int, default=30)
    ap.add_argument("--probes-per-img", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    rng = np.random.default_rng(args.seed)
    model = YOLO(args.model)
    model.model.model[-1].max_det = 1000

    gt_data = we.load_visdrone_gt(args.gt)
    items = sorted(gt_data.items(), key=lambda kv: -len(kv[1]["gt"]))
    dense = [(k, v) for k, v in items if len(v["gt"]) >= 150][:args.n_dense]
    if len(dense) < args.n_dense:
        dense += [(k, v) for k, v in items
                  if 100 <= len(v["gt"]) < 150][:args.n_dense - len(dense)]
    sparse = [(k, v) for k, v in reversed(items)
              if 5 <= len(v["gt"]) < 50][:args.n_sparse]

    rows = []
    for cohort, group in (("dense", dense), ("sparse_placebo", sparse)):
        for img_id, g in group:
            ip = os.path.join(args.images, img_id + ".jpg")
            img = cv2.imread(ip)
            if img is None:
                continue
            gt = g["gt"]
            fb, fc = run(model, img, args.imgsz, args.device)
            matched_rank, _ = we.greedy_match(fb, gt, 0.5)
            probes = pick_probes(gt, matched_rank, args.probes_per_img, rng)
            for gi in probes:
                gt_box = gt[gi]
                f_conf, f_rank = best_det_for_gt(fb, fc, gt_box)
                masked = mask_image(img, gt_box)
                mb, mc = run(model, masked, args.imgsz, args.device)
                m_conf, m_rank = best_det_for_gt(mb, mc, gt_box)
                area = float((gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]))
                rows.append({
                    "cohort": cohort, "image": img_id,
                    "n_gt": len(gt), "probe_area": round(area, 1),
                    "hit300_full": int(0 <= matched_rank[gi] < 300),
                    "full_conf": round(f_conf, 4), "full_rank": f_rank,
                    "masked_conf": round(m_conf, 4), "masked_rank": m_rank,
                    "dconf": round(m_conf - f_conf, 4),
                })
            print(f"{cohort} {img_id} n_gt={len(gt)} probes={len(probes)}",
                  flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # summary
    for cohort in ("dense", "sparse_placebo"):
        rs = [r for r in rows if r["cohort"] == cohort]
        if not rs:
            continue
        d = np.array([r["dconf"] for r in rs])
        miss = np.array([r["dconf"] for r in rs if not r["hit300_full"]])
        hit = np.array([r["dconf"] for r in rs if r["hit300_full"]])
        print(f"\n[{cohort}] probes={len(rs)}  mean dconf={d.mean():+.4f} "
              f"(median {np.median(d):+.4f})")
        if len(miss):
            print(f"  missed@300 probes: mean dconf={miss.mean():+.4f}, "
                  f"newly found (conf>0 after mask): "
                  f"{(np.array([r['masked_conf'] for r in rs if not r['hit300_full']]) > 0.05).mean():.2%}")
        if len(hit):
            print(f"  hit@300 probes:    mean dconf={hit.mean():+.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
