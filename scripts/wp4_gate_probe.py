"""Trained-head decision gate (inference-only, no training).

Greenlight a learned density->budget head ONLY if it can move a metric the
constant high cap cannot. AP is provably monotone non-decreasing in max_det
(extra ranked detections only extend the PR tail), so a smaller budget never
wins on AP -> the dense-AP-oracle axis is empty by construction. The only
surviving axes for a down-gating head are on SPARSE images:
  (1) does constant max_det=1000 COST deployed precision on sparse images
      (vs 300)? If sparse ndet@deploy-conf << 300, the extra budget only adds
      sub-threshold decodes -> precision unchanged -> axis dead.
  (2) is decode latency a non-negligible fraction of inference, so a budget
      controller has a real latency-Pareto to exploit?

Measures both on the sparse subset (GT<50) + a latency profile. Prints a
GATE verdict.

Usage: python wp4_gate_probe.py --weights <pt> --dataset {visdrone,sku}
  --imgsz N --device D
"""
import argparse
import os
import time

import numpy as np

import wp1_eval as we

ROOT = os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_sparse_yaml(dataset, gt, img_dir, names, hi=50):
    ids = [f"{k}.jpg" for k, v in gt.items() if len(v["gt"]) < hi]
    lst = os.path.join(ROOT, "data", f"_sparse_{dataset}.txt")
    with open(lst, "w", encoding="utf-8") as f:
        f.write("\n".join(os.path.join(img_dir, i) for i in ids) + "\n")
    yml = os.path.join(ROOT, "configs", f"_sparse_{dataset}.yaml")
    with open(yml, "w", encoding="utf-8") as f:
        f.write(f"path: {ROOT}\nval: {lst}\ntrain: {lst}\nnames:\n")
        for k, v in names.items():
            f.write(f"  {k}: {v}\n")
    return yml, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset", choices=["visdrone", "sku"], required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    if args.dataset == "visdrone":
        gt = we.load_visdrone_gt(os.path.join(
            ROOT, "data", "VisDrone2019-DET-val", "annotations"))
        img_dir = os.path.join(ROOT, "data", "VisDrone2019-DET-val", "images")
        names = {i: n for i, n in enumerate(
            ["pedestrian", "people", "bicycle", "car", "van", "truck",
             "tricycle", "awning-tricycle", "bus", "motor"])}
    else:
        gt = we.load_sku_gt(os.path.join(
            ROOT, "data", "SKU110K_fixed", "annotations",
            "annotations_test.csv"))
        img_dir = os.path.join(ROOT, "data", "SKU110K_fixed", "images")
        names = {0: "object"}

    yml, ids = build_sparse_yaml(args.dataset, gt, img_dir, names)
    print(f"sparse subset (GT<50): {len(ids)} images")

    from ultralytics import YOLO

    # (1) sparse-subset AP + precision at 300 vs 1000
    print("\n=== sparse-subset metrics vs budget ===")
    print(f"{'max_det':>8} {'mAP50':>8} {'mAP':>8} {'precision':>10} "
          f"{'recall':>8}")
    for k in (300, 1000):
        m = YOLO(args.weights)
        m.model.model[-1].max_det = k
        r = m.val(data=yml, imgsz=args.imgsz, max_det=k, device=args.device,
                  verbose=False, plots=False, conf=0.001)
        b = r.box
        print(f"{k:>8} {b.map50:>8.4f} {b.map:>8.4f} {b.mp:>10.4f} "
              f"{b.mr:>8.4f}", flush=True)

    # mean #det above deploy thresholds on sparse images (from a maxdet-1000 run)
    m = YOLO(args.weights)
    m.model.model[-1].max_det = 1000
    files = [os.path.join(img_dir, i) for i in ids[:200]]
    nd25, nd10 = [], []
    for f in files:
        res = m.predict(f, imgsz=args.imgsz, conf=0.001, max_det=1000,
                        device=args.device, verbose=False)[0]
        c = res.boxes.conf.cpu().numpy()
        nd25.append((c >= 0.25).sum())
        nd10.append((c >= 0.10).sum())
    print(f"\nsparse mean #det @conf>=0.25: {np.mean(nd25):.1f}  "
          f"@conf>=0.10: {np.mean(nd10):.1f}  (cap=300)")

    # (2) latency profile: total predict ms/img at 300 vs 1000
    print("\n=== latency profile (decode cost of relaxing budget) ===")
    lat = {}
    for k in (300, 1000):
        m = YOLO(args.weights)
        m.model.model[-1].max_det = k
        for f in files[:10]:  # warmup
            m.predict(f, imgsz=args.imgsz, conf=0.001, max_det=k,
                      device=args.device, verbose=False)
        t0 = time.time()
        for f in files:
            m.predict(f, imgsz=args.imgsz, conf=0.001, max_det=k,
                      device=args.device, verbose=False)
        lat[k] = (time.time() - t0) / len(files) * 1000
        print(f"max_det={k}: {lat[k]:.2f} ms/img")
    delta = (lat[1000] - lat[300]) / lat[300] * 100
    print(f"latency delta 300->1000: {delta:+.1f}%")

    # GATE verdict
    print("\n=== GATE VERDICT ===")
    sparse_cap_binds = np.mean(nd25) >= 300 * 0.5
    latency_matters = abs(delta) >= 10.0
    print(f"  sparse deploy-cap binds (ndet@.25 >= 150)? {sparse_cap_binds}")
    print(f"  decode latency non-negligible (>=10% delta)? {latency_matters}")
    if sparse_cap_binds or latency_matters:
        print("  -> a B-axis MAY exist; inspect sparse-precision delta above.")
    else:
        print("  -> NEITHER axis realizable: constant high cap costs nothing "
              "on sparse images, decode latency negligible. B (trained head) "
              "is DEAD. Ship route A.")


if __name__ == "__main__":
    main()
