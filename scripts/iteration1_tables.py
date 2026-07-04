"""Cache-based result-table artifacts:
  (1) recovery-mechanism comparison table (intervention vs intervention),
  (7) per-class AP on VisDrone at K=300 vs K=1000,
  (8) gate hyperparameter sensitivity sweep.
All from existing prediction caches (no new inference).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(ROOT, "scripts"))
import wp1_eval as we
import wp4_ap_bootstrap as ab

ROOT = os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(ROOT, "experiments")
VD_GT = os.path.join(ROOT, "data", "VisDrone2019-DET-val", "annotations")
SKU_GT = os.path.join(ROOT, "data", "SKU110K_fixed", "annotations",
                      "annotations_test.csv")
VD_NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
            "tricycle", "awning-tricycle", "bus", "motor"]


def _ap_allpoint(per_det, ngt):
    if ngt == 0 or not per_det:
        return 0.0
    per_det = sorted(per_det, key=lambda z: -z[0])
    tp = np.array([d[1] for d in per_det])
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / ngt, ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0], rec, [rec[-1]]])
    mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def ap50(preds, gt_data, ids, k, n_classes):
    """class-mean all-point AP@0.5 over top-k ranked preds."""
    aps = []
    for c in range(n_classes):
        per_det, ngt = [], 0
        for img in ids:
            g = gt_data.get(img)
            if g is None:
                continue
            m = g["gt_cls"] == c
            ngt += int(m.sum())
            p = preds.get(img)
            if p is None or len(p) == 0:
                continue
            p = p[:k]
            pm = p if n_classes == 1 else p[p[:, 5].astype(int) == c]
            if len(pm) == 0:
                continue
            gtb = g["gt"][m] if n_classes > 1 else g["gt"]
            order = pm[:, 4].argsort()[::-1]
            pm = pm[order]
            taken = np.zeros(len(gtb), bool)
            ious = we.iou_matrix(pm[:, :4], gtb)
            for r in range(len(pm)):
                tp = 0
                if taken.size:
                    row = np.where(~taken, ious[r], -1)
                    j = int(row.argmax())
                    if row[j] >= 0.5:
                        taken[j] = True
                        tp = 1
                per_det.append((pm[r, 4], tp))
        if ngt:
            aps.append(_ap_allpoint(per_det, ngt))
    return float(np.mean(aps)) if aps else 0.0


def recall_at(preds, gt_data, ids, k, conf_floor, class_aware):
    """class-agnostic/aware recall among top-k preds with conf>=floor."""
    tp = gt = 0
    for img in ids:
        g = gt_data.get(img)
        if g is None:
            continue
        gtb, gcls = g["gt"], g["gt_cls"]
        gt += len(gtb)
        p = preds.get(img)
        if p is None or len(p) == 0:
            continue
        p = p[:k]
        p = p[p[:, 4] >= conf_floor]
        mr, _ = we.greedy_match(p[:, :4], gtb, 0.5,
                                pred_cls=(p[:, 5].astype(int) if class_aware else None),
                                gt_cls=(gcls if class_aware else None))
        tp += int((mr >= 0).sum())
    return tp / max(gt, 1)


def dense_ids(gt_data, preds, thr=150):
    return [k for k, v in gt_data.items()
            if len(v["gt"]) >= thr and k in preds]


def recovery_table():
    print("\n" + "=" * 70 + "\n(1) RECOVERY-MECHANISM COMPARISON (dense subset GT>=150)\n" + "=" * 70)
    # SKU
    sku_gt = we.load_sku_gt(SKU_GT)
    e2e = ab.load_preds(os.path.join(EXP, "wp1_sku", "preds_sku_test_ft.jsonl"))
    o2m = ab.load_preds(os.path.join(EXP, "wp1_sku", "preds_sku_test_ft_o2m.jsonl"))
    rtd = ab.load_preds(os.path.join(EXP, "wp1_sku", "preds_sku_rtdetr.jsonl"))
    ids = dense_ids(sku_gt, e2e)
    print(f"\nSKU-110K dense (n={len(ids)}):")
    print(f"{'method':>34} {'AP@.5':>7} {'R@.25':>7} {'recov@.25':>10}")
    rows = [
        ("YOLO26 e2e, cap K=300 (default)", e2e, 300, False),
        ("YOLO26 e2e, cap K=1000 (our repair)", e2e, 1000, False),
        ("YOLO26 o2m + NMS (iou0.5, agn.)", o2m, 1000, False),
        ("RT-DETR (hard 300-query)", rtd, 1000, False),
    ]
    base = recall_at(e2e, sku_gt, ids, 300, 0.25, False)
    for name, pr, k, ca in rows:
        a = ap50(pr, sku_gt, ids, k, 1)
        r = recall_at(pr, sku_gt, ids, k, 0.25, ca)
        print(f"{name:>34} {a:>7.3f} {r:>7.3f} {r-base:>+10.3f}")

    # VisDrone (class-aware, 10 cls)
    vd_gt = we.load_visdrone_gt(VD_GT)
    e2ev = ab.load_preds(os.path.join(EXP, "wp1_ft", "preds_visdrone_ft_s.jsonl"))
    o2mv = ab.load_preds(os.path.join(EXP, "wp1_ft", "preds_ft_s_o2m_nms05_ag.jsonl"))
    idsv = dense_ids(vd_gt, e2ev)
    print(f"\nVisDrone dense (n={len(idsv)}, class-aware):")
    print(f"{'method':>34} {'AP@.5':>7} {'R@.25':>7} {'recov@.25':>10}")
    basev = recall_at(e2ev, vd_gt, idsv, 300, 0.25, True)
    for name, pr, k, ca in [
        ("YOLO26 e2e, cap K=300 (default)", e2ev, 300, True),
        ("YOLO26 e2e, cap K=1000 (our repair)", e2ev, 1000, True),
        ("YOLO26 o2m + NMS (iou0.5, agn.)", o2mv, 1000, True)]:
        a = ap50(pr, vd_gt, idsv, k, 10)
        r = recall_at(pr, vd_gt, idsv, k, 0.25, ca)
        print(f"{name:>34} {a:>7.3f} {r:>7.3f} {r-basev:>+10.3f}")


def per_class_ap():
    print("\n" + "=" * 70 + "\n(7) PER-CLASS AP@0.5 on VisDrone, K=300 vs K=1000 (dense subset)\n" + "=" * 70)
    vd_gt = we.load_visdrone_gt(VD_GT)
    preds = ab.load_preds(os.path.join(EXP, "wp1_ft", "preds_visdrone_ft_s.jsonl"))
    ids = dense_ids(vd_gt, preds)
    # compute single-class AP per class by masking
    print(f"{'class':>16} {'AP@300':>8} {'AP@1000':>8} {'delta':>7}")
    for c in range(10):
        # build a 1-class view
        def ap_c(k):
            per_det, ngt = [], 0
            for img in ids:
                g = vd_gt[img]
                m = g["gt_cls"] == c
                ngt += int(m.sum())
                p = preds.get(img)
                if p is None or len(p) == 0:
                    continue
                p = p[:k]
                pm = p[p[:, 5].astype(int) == c]
                if len(pm) == 0:
                    continue
                ious = we.iou_matrix(pm[:, :4], g["gt"][m])
                taken = np.zeros(int(m.sum()), bool)
                for r in range(len(pm)):
                    tp = 0
                    if taken.size:
                        row = np.where(~taken, ious[r], -1)
                        j = int(row.argmax())
                        if row[j] >= 0.5:
                            taken[j] = True
                            tp = 1
                    per_det.append((pm[r, 4], tp))
            if ngt == 0 or not per_det:
                return 0.0
            per_det.sort(key=lambda z: -z[0])
            tp = np.array([d[1] for d in per_det])
            ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
            rec, prec = ctp / ngt, ctp / np.maximum(ctp + cfp, 1e-9)
            mrec = np.concatenate([[0], rec, [rec[-1]]])
            mpre = np.concatenate([[0], prec, [0]])
            for i in range(len(mpre) - 1, 0, -1):
                mpre[i - 1] = max(mpre[i - 1], mpre[i])
            idx = np.where(mrec[1:] != mrec[:-1])[0]
            return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
        a3, a1 = ap_c(300), ap_c(1000)
        print(f"{VD_NAMES[c]:>16} {a3:>8.3f} {a1:>8.3f} {a1-a3:>+7.3f}")


def gate_sweep():
    print("\n" + "=" * 70 + "\n(8) GATE HYPERPARAMETER SENSITIVITY (SKU dense >=300 bucket recall)\n" + "=" * 70)
    sku_gt = we.load_sku_gt(SKU_GT)
    preds = ab.load_preds(os.path.join(EXP, "wp1_sku", "preds_sku_test_ft.jsonl"))
    ids = [k for k, v in sku_gt.items() if len(v["gt"]) >= 300 and k in preds]
    print(f"dense >=300 images: {len(ids)}")

    def policy_recall(tau, edges, tiers):
        tp = gt = slots = 0
        for img in ids:
            g = sku_gt[img]
            gtb = g["gt"]
            gt += len(gtb)
            p = preds.get(img)
            if p is None:
                continue
            n = int((p[:, 4] >= tau).sum())
            k = tiers[0] if n <= edges[0] else (tiers[1] if n <= edges[1] else tiers[2])
            slots += k
            mr, _ = we.greedy_match(p[:k, :4], gtb, 0.5)
            tp += int((mr >= 0).sum())
        return tp / max(gt, 1), slots / len(ids)

    print("\n-- vary conf proxy tau (edges 100/200, tiers 300/600/1000) --")
    for tau in (0.05, 0.10, 0.25):
        r, s = policy_recall(tau, (100, 200), (300, 600, 1000))
        print(f"  tau={tau}: recall={r:.3f} meanK={s:.0f}")
    print("-- vary budget tiers (tau 0.1, edges 100/200) --")
    for tiers in ((300, 600, 1000), (300, 600, 1500), (300, 1000, 2000)):
        r, s = policy_recall(0.1, (100, 200), tiers)
        print(f"  tiers={tiers}: recall={r:.3f} meanK={s:.0f}")
    print("-- vary bucket edges (tau 0.1, tiers 300/600/1000) --")
    for edges in ((100, 200), (150, 250), (80, 160)):
        r, s = policy_recall(0.1, edges, (300, 600, 1000))
        print(f"  edges={edges}: recall={r:.3f} meanK={s:.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all")
    args = ap.parse_args()
    if args.which in ("all", "1"):
        recovery_table()
    if args.which in ("all", "7"):
        per_class_ap()
    if args.which in ("all", "8"):
        gate_sweep()
