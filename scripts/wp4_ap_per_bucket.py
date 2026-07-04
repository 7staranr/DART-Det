"""Per-density-bucket AP vs budget.

Builds val image-lists per density bucket (image GT count), runs ultralytics
val at each max_det, records mAP50 / mAP50-95 / precision. Shows the budget
effect is monotone and bucket-graded (flat on sparse, rising on dense), with
precision held constant — the deployable AP story.

Usage:
  python wp4_ap_per_bucket.py --weights <pt> --dataset {visdrone,sku}
    --imgsz N --device D --budgets 300,600,1000
"""
import argparse
import os

import wp1_eval as we

ROOT = os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUCKETS = [(50, 100), (100, 150), (150, 300), (300, 10**9)]
BNAMES = ["50-100", "100-150", "150-300", ">=300"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset", choices=["visdrone", "sku"], required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="0")
    ap.add_argument("--budgets", default="300,600,1000")
    args = ap.parse_args()
    budgets = [int(x) for x in args.budgets.split(",")]

    if args.dataset == "visdrone":
        gt = we.load_visdrone_gt(os.path.join(
            ROOT, "data", "VisDrone2019-DET-val", "annotations"))
        img_dir = os.path.join(ROOT, "data", "VisDrone2019-DET-val", "images")
        names = {i: n for i, n in enumerate(
            ["pedestrian", "people", "bicycle", "car", "van", "truck",
             "tricycle", "awning-tricycle", "bus", "motor"])}
        work = os.path.join(ROOT, "data", "VisDrone2019-DET-val")
    else:
        gt = we.load_sku_gt(os.path.join(
            ROOT, "data", "SKU110K_fixed", "annotations",
            "annotations_test.csv"))
        img_dir = os.path.join(ROOT, "data", "SKU110K_fixed", "images")
        names = {0: "object"}
        work = os.path.join(ROOT, "data", "SKU110K_fixed")

    from ultralytics import YOLO

    for lo, hi, bn in [(b[0], b[1], n) for b, n in zip(BUCKETS, BNAMES)]:
        ids = [f"{k}.jpg" for k, v in gt.items() if lo <= len(v["gt"]) < hi]
        if len(ids) < 5:
            print(f"\n## bucket {bn}: only {len(ids)} imgs, skip")
            continue
        lst = os.path.join(work, f"bucket_{bn.replace('>=','ge')}.txt")
        with open(lst, "w", encoding="utf-8") as f:
            f.write("\n".join(os.path.join(img_dir, i) for i in ids) + "\n")
        yml = os.path.join(ROOT, "configs",
                           f"_apbucket_{args.dataset}.yaml")
        with open(yml, "w", encoding="utf-8") as f:
            f.write(f"path: {ROOT}\nval: {lst}\ntrain: {lst}\nnames:\n")
            for k, v in names.items():
                f.write(f"  {k}: {v}\n")
        print(f"\n## bucket {bn} ({len(ids)} imgs)")
        print(f"{'max_det':>8} {'mAP50':>8} {'mAP50-95':>9} {'P':>7} {'R':>7}")
        for k in budgets:
            m = YOLO(args.weights)
            m.model.model[-1].max_det = k
            r = m.val(data=yml, imgsz=args.imgsz, max_det=k,
                      device=args.device, verbose=False, plots=False,
                      conf=0.001)
            b = r.box
            print(f"{k:>8} {b.map50:>8.4f} {b.map:>9.4f} {b.mp:>7.4f} "
                  f"{b.mr:>7.4f}", flush=True)


if __name__ == "__main__":
    main()
