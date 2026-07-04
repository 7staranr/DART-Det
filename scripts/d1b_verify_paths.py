"""D1b: corrected verification of budget relaxation and dual-path switching.

Facts established by source reading (ultralytics 8.4.64):
  - Detect head class attr max_det=300 drives the e2e topk (head.py:81,231).
  - predict(max_det=...) only truncates AFTER the head topk -> must raise the
    head module attr to relax the budget.
  - end2end is a DetectionModel property setter -> model.model.end2end=False
    switches the head to one2many dense output + NMS branch (nms.py:66 picks
    branch by output shape).
  - fuse() with end2end=True REMOVES the one2many branch -> separate model
    instances per path; switch BEFORE first predict.

Checks:
 A. head parameter inventory: does the checkpoint contain one2many weights?
 B. e2e budget relaxation: head.max_det=2000 -> n_det > 300?
 C. truncation equivalence with relaxed cap: top-300(K=2000) == default-300?
 D. o2m+NMS path via model.model.end2end=False on a FRESH instance.
"""
import os
ROOT = os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from ultralytics import YOLO

IMG = os.path.join(ROOT, "data", "VisDrone2019-DET-val", "images", "0000295_02400_d_0000033.jpg")
WN = os.path.join(ROOT, "weights", "yolo26n.pt")


def boxes_array(res):
    b = res.boxes
    arr = np.concatenate(
        [b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None],
         b.cls.cpu().numpy()[:, None]], axis=1)
    return arr[arr[:, 4].argsort()[::-1]]


print("=== A. head inventory ===")
m = YOLO(WN)
head = m.model.model[-1]
print("head class:", type(head).__name__)
print("head.end2end:", getattr(head, "end2end", None))
print("head.max_det:", getattr(head, "max_det", None))
children = dict(head.named_children())
print("head children:", list(children.keys()))
n_params = {k: sum(p.numel() for p in v.parameters()) for k, v in children.items()}
print("param counts:", n_params)

print("\n=== baseline e2e (default cap 300) ===")
r_def = m.predict(IMG, imgsz=1280, conf=0.001, device=0, verbose=False)[0]
a_def = boxes_array(r_def)
print("n_det:", len(a_def))

print("\n=== B. e2e relaxed cap ===")
m2 = YOLO(WN)
m2.model.model[-1].max_det = 2000
r_relax = m2.predict(IMG, imgsz=1280, conf=0.001, max_det=2000,
                     device=0, verbose=False)[0]
a_relax = boxes_array(r_relax)
print("n_det with head.max_det=2000:", len(a_relax))

print("\n=== C. truncation equivalence ===")
k = min(300, len(a_def), len(a_relax))
# compare sets: sort both by conf desc and compare coordinates
diff = np.abs(a_def[:k, :5] - a_relax[:k, :5]).max()
print(f"max|diff| top-{k}: {diff:.6f}")

print("\n=== D. one2many + NMS path (fresh instance) ===")
m3 = YOLO(WN)
try:
    m3.model.end2end = False
    print("model.end2end now:", m3.model.end2end)
    r_o2m = m3.predict(IMG, imgsz=1280, conf=0.001, max_det=2000,
                       device=0, verbose=False)[0]
    a_o2m = boxes_array(r_o2m)
    print("o2m n_det (conf>=0.001, NMS, max_det=2000):", len(a_o2m))
    for t in (0.25, 0.10, 0.05):
        print(f"  o2m dets conf>={t}: {(a_o2m[:, 4] >= t).sum()}  | "
              f"e2e-relaxed: {(a_relax[:, 4] >= t).sum()}")
except Exception as e:
    import traceback
    traceback.print_exc()

print("\nD1b complete.")
