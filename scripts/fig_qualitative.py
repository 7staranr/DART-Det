"""Qualitative TP/FP/FN figure, cap-300 vs cap-1000, on a dense image.

Runs the finetuned model on one dense VisDrone and one dense SKU image, matches
detections to GT (greedy IoU 0.5), and draws color-coded boxes:
  green = TP, red = FP, blue = FN (missed GT), at K=300 and K=1000 side by side.
Makes budget truncation visually legible.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(ROOT, "scripts"))
import wp1_eval as we

ROOT = os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "paper", "figures")
GREEN, RED, BLUE = (0, 180, 0), (0, 0, 230), (230, 60, 0)  # BGR


def draw(img, boxes, color, thick=2):
    for b in boxes:
        x1, y1, x2, y2 = [int(v) for v in b[:4]]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    return img


def panel(model, img_path, gt, gcls, k, imgsz, device, class_aware):
    import torch  # noqa
    model.model.model[-1].max_det = k   # relax the e2e head topk (not just predict arg)
    res = model.predict(img_path, imgsz=imgsz, conf=0.001, max_det=k,
                        device=device, verbose=False)[0]
    b = res.boxes
    xyxy = b.xyxy.cpu().numpy()
    conf = b.conf.cpu().numpy()
    cls = b.cls.cpu().numpy().astype(int)
    order = conf.argsort()[::-1][:k]
    xyxy, cls = xyxy[order], cls[order]
    # deploy threshold for display: keep conf>=0.25 (what a user sees)
    keep = conf[order] >= 0.25
    xyxy_d, cls_d = xyxy[keep], cls[keep]
    mr, _ = we.greedy_match(xyxy_d, gt, 0.5,
                            pred_cls=(cls_d if class_aware else None),
                            gt_cls=(gcls if class_aware else None))
    tp = xyxy_d[[i for i in range(len(xyxy_d))
                 if i in set(mr[mr >= 0])]] if len(xyxy_d) else np.zeros((0, 4))
    # simpler: a det is TP if it is the matcher of some GT
    matched_pred_idx = set(int(r) for r in mr if r >= 0)
    tp = np.array([xyxy_d[i] for i in range(len(xyxy_d))
                   if i in matched_pred_idx]).reshape(-1, 4)
    fp = np.array([xyxy_d[i] for i in range(len(xyxy_d))
                   if i not in matched_pred_idx]).reshape(-1, 4)
    fn = gt[mr < 0] if len(gt) else np.zeros((0, 4))
    img = cv2.imread(img_path)
    draw(img, tp, GREEN)
    draw(img, fp, RED, 1)
    draw(img, fn, BLUE)
    n_gt = len(gt)
    cv2.putText(img, f"K={k}: recall {len(tp)}/{n_gt}  (FN={len(fn)})",
                (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 4)
    cv2.putText(img, f"K={k}: recall {len(tp)}/{n_gt}  (FN={len(fn)})",
                (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    return img


def main():
    from ultralytics import YOLO
    os.makedirs(OUT, exist_ok=True)
    # densest VisDrone val image (317 GT) + a dense SKU test image
    jobs = [
        ("sku",
         os.path.join(ROOT, "runs", "ft_sku110k_yolo26n_1024", "weights", "best.pt"),
         os.path.join(ROOT, "data", "SKU110K_fixed", "images", "test_129.jpg"),
         we.load_sku_gt(os.path.join(ROOT, "data", "SKU110K_fixed",
                                     "annotations", "annotations_test.csv")),
         1024, False),
    ]
    for tag, w, ip, gtmap, imgsz, ca in jobs:
        stem = os.path.splitext(os.path.basename(ip))[0]
        g = gtmap[stem]
        gt, gcls = g["gt"], g["gt_cls"]
        model = YOLO(w)
        p300 = panel(model, ip, gt, gcls, 300, imgsz, 0, ca)
        p1000 = panel(model, ip, gt, gcls, 1000, imgsz, 0, ca)
        h = max(p300.shape[0], p1000.shape[0])
        combo = np.hstack([p300, np.full((h, 8, 3), 255, np.uint8), p1000])
        out = os.path.join(OUT, f"fig_qualitative_{tag}.png")
        cv2.imwrite(out, combo)
        print("wrote", out, combo.shape)


if __name__ == "__main__":
    main()
