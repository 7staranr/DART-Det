"""D1 verification: YOLO26 inference semantics that WP1 design depends on.

Checks, on one dense VisDrone image:
 1. yolo26n.pt loads; CUDA works.
 2. e2e (one-to-one) path: does max_det>300 actually return >300 ranked dets?
 3. truncation equivalence: top-300 of a max_det=1000 run == max_det=300 run.
 4. end2end=False switches to one-to-many + NMS path (H3 control).
 5. count detections above deployment thresholds on a dense image.
"""
import os
ROOT = os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys

import numpy as np
import torch
from ultralytics import YOLO

IMG = os.path.join(ROOT, "data", "VisDrone2019-DET-val", "images", "0000295_02400_d_0000033.jpg")  # densest val image: 317 GT


def boxes_array(res):
    b = res.boxes
    arr = np.concatenate(
        [b.xyxy.cpu().numpy(),
         b.conf.cpu().numpy()[:, None],
         b.cls.cpu().numpy()[:, None]], axis=1)
    return arr[arr[:, 4].argsort()[::-1]]


def main():
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    import ultralytics
    print("ultralytics", ultralytics.__version__)

    model = YOLO("yolo26n.pt")
    print("model loaded:", type(model.model).__name__)
    print("end2end attr:", getattr(model.model, "end2end", None))

    img = sys.argv[1] if len(sys.argv) > 1 else IMG

    # 2: e2e with different max_det
    r300 = model.predict(img, imgsz=1280, conf=0.001, max_det=300,
                         device=0, verbose=False)[0]
    r1000 = model.predict(img, imgsz=1280, conf=0.001, max_det=1000,
                          device=0, verbose=False)[0]
    a300, a1000 = boxes_array(r300), boxes_array(r1000)
    print(f"\n[e2e] n_det max_det=300 : {len(a300)}")
    print(f"[e2e] n_det max_det=1000: {len(a1000)}")

    # 3: truncation equivalence
    k = min(len(a300), len(a1000), 300)
    diff = np.abs(a300[:k, :5] - a1000[:k, :5]).max() if k else float("nan")
    print(f"[e2e] truncation equivalence max|diff| over top-{k}: {diff:.6f}")

    # 4: one-to-many + NMS path
    try:
        rnms = model.predict(img, imgsz=1280, conf=0.001, max_det=1000,
                             end2end=False, device=0, verbose=False)[0]
        anms = boxes_array(rnms)
        print(f"\n[o2m+NMS] n_det max_det=1000: {len(anms)} (end2end=False OK)")
    except Exception as e:
        print(f"\n[o2m+NMS] end2end=False FAILED: {e}")

    # 5: deployment-threshold counts
    for t in (0.25, 0.10, 0.05):
        print(f"[e2e] dets with conf>={t}: {(a1000[:, 4] >= t).sum()}")

    print("\nD1 verification complete.")


if __name__ == "__main__":
    main()
