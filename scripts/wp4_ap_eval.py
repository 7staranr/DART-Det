"""AP test: does relaxing the end-to-end budget raise or lower COCO AP?

A natural concern is that the adaptive budget is "recall at any cost" (huge
FP/img). But FP/img was measured at conf=0.001 (the oracle list). COCO AP
integrates precision over all thresholds; the fixed 300-cap *truncates* the PR
curve, so lifting max_det can only ADD recall at the low-precision tail ->
AP should be monotonic non-decreasing in max_det. This script measures it with
ultralytics' own validator (proper AP@0.5 and AP@0.5:0.95).

For the e2e head, max_det is a module attr (head.py:81); we set it before each
val() so the topk actually relaxes.

Usage: python wp4_ap_eval.py --weights <pt> --data <yaml> --imgsz N --device D
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="0")
    ap.add_argument("--budgets", default="300,600,1000")
    args = ap.parse_args()

    from ultralytics import YOLO

    budgets = [int(x) for x in args.budgets.split(",")]
    print(f"{'max_det':>8} {'mAP50':>8} {'mAP50-95':>9} {'precision':>10} "
          f"{'recall':>8}")
    for k in budgets:
        model = YOLO(args.weights)          # fresh each time (clean state)
        model.model.model[-1].max_det = k   # relax the e2e topk
        m = model.val(data=args.data, imgsz=args.imgsz, max_det=k,
                      device=args.device, verbose=False, plots=False,
                      conf=0.001)
        b = m.box
        print(f"{k:>8} {b.map50:>8.4f} {b.map:>9.4f} {b.mp:>10.4f} "
              f"{b.mr:>8.4f}", flush=True)


if __name__ == "__main__":
    main()
