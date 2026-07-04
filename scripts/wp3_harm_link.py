"""WP3 harm link: do assignment-unstable GTs end up worse detected?

Joins per-GT flip count (across the 20-epoch probe training) with the FINAL
probe model's detection outcome on the same images (in-sample by design:
the question is whether training-time instability of a GT predicts its final
detectability, holding crowding/size strata fixed).

Output: detection rate (hit within top-300, IoU 0.5) by flip-count bin,
overall and within (crowding x size) strata; plus a within-image paired
contrast: among GTs of the same image and size class, unstable (top-tertile
flips) vs stable (bottom-tertile).
"""
import os
import re

import numpy as np

import wp1_eval as we

ROOT = os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DYN = os.path.join(ROOT, "runs", "wp3_dynamics")
WEIGHTS = os.path.join(ROOT, "runs", "wp3_probe_train", "weights", "last.pt")


def per_gt_flips():
    files = {}
    for fn in os.listdir(DYN):
        m = re.match(r"probe_epoch(\d+)\.npz", fn)
        if m:
            files[int(m.group(1))] = os.path.join(DYN, fn)
    eps = {}
    for ep in sorted(files):
        rows = np.load(files[ep])["rows"]
        eps[ep] = {(int(r[1]), int(r[2]), int(r[3])): int(r[4])
                   for r in rows}
    keys = set.intersection(*(set(d.keys()) for d in eps.values()))
    epochs = sorted(eps)
    flips = {}
    for k in keys:
        seq = [eps[e][k] for e in epochs]
        flips[k] = sum(1 for a, b in zip(seq[:-1], seq[1:])
                       if a != b and a >= 0 and b >= 0)
    return flips


def main():
    import torch
    from ultralytics import YOLO

    flips = per_gt_flips()
    print(f"per-GT flip counts: {len(flips)}")

    with open(os.path.join(DYN, "probe_ids.txt")) as f:
        ids = sorted(ln.strip() for ln in f if ln.strip())
    gt_all = we.load_visdrone_gt(
        os.path.join(ROOT, "data", "VisDrone2019-DET-train", "annotations"))

    model = YOLO(WEIGHTS)
    model.model.model[-1].max_det = 1000

    recs = []
    for pos, img_id in enumerate(ids):
        ip = os.path.join(ROOT, "data", "VisDrone2019-DET-train", "images",
                          img_id + ".jpg")
        gt = gt_all[img_id]["gt"]
        res = model.predict(ip, imgsz=1280, conf=0.001, max_det=1000,
                            device=1, verbose=False)[0]
        b = res.boxes
        xyxy = b.xyxy.cpu().numpy()
        conf = b.conf.cpu().numpy()
        order = conf.argsort()[::-1]
        mr, _ = we.greedy_match(xyxy[order], gt, 0.5)

        ious = we.iou_matrix(gt, gt)
        np.fill_diagonal(ious, 0)
        max_iou = ious.max(1) if len(gt) > 1 else np.zeros(len(gt))
        area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
        bi, bb = divmod(pos, 4)
        for gi in range(len(gt)):
            k = (bi, bb, gi)
            if k not in flips:
                continue
            recs.append({
                "img": img_id, "n_gt": len(gt), "flips": flips[k],
                "crowd": float(max_iou[gi]),
                "small": bool(area[gi] < 32**2),
                "hit300": int(0 <= mr[gi] < 300),
            })
    print(f"joined records: {len(recs)}")

    fl = np.array([r["flips"] for r in recs])
    hit = np.array([r["hit300"] for r in recs])
    crowd = np.array([r["crowd"] for r in recs])
    small = np.array([r["small"] for r in recs])

    print("\n--- detection rate by flip-count bin ---")
    for name, sel in (("0-1 flips", fl <= 1), ("2-3", (fl >= 2) & (fl <= 3)),
                      ("4-6", (fl >= 4) & (fl <= 6)), (">=7", fl >= 7)):
        if sel.sum() > 50:
            print(f"  {name:>10}: hit300={hit[sel].mean():.3f}  n={sel.sum()}")

    print("\n--- within strata (crowding x size) ---")
    for cn, cs in (("isolated", crowd < 0.1), ("crowded", crowd >= 0.3)):
        for sn, ss in (("small", small), ("med+large", ~small)):
            sel = cs & ss
            if sel.sum() < 100:
                continue
            lo = sel & (fl <= 1)
            hi = sel & (fl >= 4)
            if lo.sum() > 50 and hi.sum() > 50:
                print(f"  {cn}/{sn}: stable(<=1) {hit[lo].mean():.3f} "
                      f"(n={lo.sum()})  vs unstable(>=4) {hit[hi].mean():.3f} "
                      f"(n={hi.sum()})  gap={hit[lo].mean() - hit[hi].mean():+.3f}")

    # within-image paired contrast (controls scene-level confounds)
    print("\n--- within-image paired contrast (same image, same size class) ---")
    from collections import defaultdict
    gaps = []
    by_img = defaultdict(list)
    for r in recs:
        by_img[(r["img"], r["small"])].append(r)
    for (_, _), rs in by_img.items():
        f = np.array([r["flips"] for r in rs])
        h = np.array([r["hit300"] for r in rs])
        if len(rs) < 6:
            continue
        t_lo, t_hi = np.quantile(f, [0.33, 0.67])
        lo, hi = h[f <= t_lo], h[f >= max(t_hi, t_lo + 1)]
        if len(lo) >= 2 and len(hi) >= 2:
            gaps.append(lo.mean() - hi.mean())
    gaps = np.array(gaps)
    rng = np.random.default_rng(0)
    boot = [rng.choice(gaps, len(gaps)).mean() for _ in range(3000)]
    lo_ci, hi_ci = np.percentile(boot, [2.5, 97.5])
    print(f"  mean(stable - unstable) detection gap = {gaps.mean():+.4f} "
          f"(95% CI [{lo_ci:+.4f}, {hi_ci:+.4f}], {len(gaps)} image-strata)")


if __name__ == "__main__":
    main()
