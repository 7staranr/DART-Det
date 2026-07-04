"""Empirical capacity / rate-budget curve: recall-at-budget R@k vs budget k on
dense subsets, for soft top-K detectors (rises with k -> recoverable RTIL) versus
the hard-query RT-DETR (flat -> no rank tail). Gives the channel framing real
empirical content."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(ROOT, "scripts"))
import wp1_eval as we
import wp4_ap_bootstrap as ab

ROOT = os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KS = [300, 400, 500, 600, 700, 800, 1000]


def recall_at_ks(preds, gt, ids, ks, ca=False):
    out = []
    for k in ks:
        tp = ngt = 0
        for img in ids:
            g = gt.get(img)
            if g is None:
                continue
            ngt += len(g["gt"])
            p = preds.get(img)
            if p is None or len(p) == 0:
                continue
            pk = p[:k]
            mr, _ = we.greedy_match(pk[:, :4], g["gt"], 0.5,
                                    pred_cls=(pk[:, 5].astype(int) if ca else None),
                                    gt_cls=(g["gt_cls"] if ca else None))
            tp += int((mr >= 0).sum())
        out.append(tp / max(ngt, 1))
    return out


def dense_ids(gt, preds, thr):
    return [k for k, v in gt.items() if len(v["gt"]) >= thr and k in preds]


vd = we.load_visdrone_gt(os.path.join(ROOT, "data", "VisDrone2019-DET-val", "annotations"))
dota = we.load_yolo_gt(os.path.join(ROOT, "data", "DOTAv1-tiled", "labels", "val"))

curves = []
# soft: YOLO26-s VisDrone dense
p = ab.load_preds(os.path.join(ROOT, "experiments", "wp1_ft", "preds_visdrone_ft_s.jsonl"))
curves.append(("YOLO26-s VisDrone (soft)", recall_at_ks(p, vd, dense_ids(vd, p, 150), KS), "-o", "#1f77b4"))
# soft: DOTA >=300
p = ab.load_preds(os.path.join(ROOT, "experiments", "wp1_ft", "preds_dota_ft.jsonl"))
curves.append(("YOLO26-n DOTA, $\\geq$300 (soft)", recall_at_ks(p, dota, dense_ids(dota, p, 300), KS), "-s", "#2ca02c"))
# hard: RT-DETR VisDrone dense
rt = os.path.join(ROOT, "experiments", "wp1_rtdetr", "preds_visdrone_rtdetr.jsonl")
if os.path.exists(rt):
    p = ab.load_preds(rt)
    curves.append(("RT-DETR VisDrone (hard)", recall_at_ks(p, vd, dense_ids(vd, p, 150), KS), "--^", "#d62728"))

fig, ax = plt.subplots(figsize=(6.2, 4.2))
for lbl, ys, st, c in curves:
    ax.plot(KS, ys, st, label=lbl, color=c, lw=1.8, ms=6)
ax.axvline(300, color="gray", ls=":", lw=1.2)
ax.text(305, ax.get_ylim()[0] + 0.01, "deployed budget $K{=}300$", fontsize=8, color="gray")
ax.set_xlabel("decode budget $k$ (cache depth)")
ax.set_ylabel("recall-at-budget $R@k$ (dense subset)")
ax.set_title("Empirical capacity curve: recoverable recall vs budget")
ax.legend(fontsize=8.5, loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
out = os.path.join(ROOT, "paper", "figures", "fig_capacity_curve.pdf")
plt.savefig(out, bbox_inches="tight")
print("wrote", out)
for lbl, ys, _, _ in curves:
    print(lbl, [round(y, 3) for y in ys])
