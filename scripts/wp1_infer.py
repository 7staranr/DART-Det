"""WP1 inference runner: cache ranked predictions for offline AR@k analysis.

Runs a YOLO26 (or any ultralytics) model over a dataset's images once with a
large max_det and near-zero confidence threshold, then saves ALL ranked
predictions per image to a .jsonl cache. Because the end-to-end head selects
top-k by score, AR@300 computed by truncating this cache is identical to
running inference with max_det=300.

Usage:
  python wp1_infer.py --model yolo26n.pt --images <dir> --out <preds.jsonl>
                      --imgsz 1280 --max-det 1000 [--device 0]
"""
import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", default=None,
                    help="directory of images (all files)")
    ap.add_argument("--list", default=None,
                    help="txt file with absolute image paths (one per line)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--max-det", type=int, default=1000)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--o2m", action="store_true",
                    help="use one-to-many head + NMS instead of e2e head")
    ap.add_argument("--nms-iou", type=float, default=0.7,
                    help="NMS IoU threshold for the o2m path")
    ap.add_argument("--agnostic-nms", action="store_true",
                    help="class-agnostic NMS for the o2m path")
    args = ap.parse_args()

    if "rtdetr" in os.path.basename(args.model).lower() or \
            "rtdetr" in args.model.lower():
        from ultralytics import RTDETR
        model = RTDETR(args.model)
        # RT-DETR budget is architecturally hard (300 decoder queries):
        # max_det relaxation impossible; o2m switch does not exist.
    else:
        from ultralytics import YOLO

        model = YOLO(args.model)
        if args.o2m:
            # switch BEFORE first predict (fuse would strip o2m)
            model.model.end2end = False
        else:
            # head module attr drives the e2e topk
            model.model.model[-1].max_det = args.max_det
    if args.list:
        with open(args.list, "r", encoding="utf-8") as f:
            img_files = sorted(ln.strip() for ln in f if ln.strip())
    else:
        img_files = sorted(
            os.path.join(args.images, f)
            for f in os.listdir(args.images)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        )
    print(f"model={args.model} images={len(img_files)} imgsz={args.imgsz} "
          f"max_det={args.max_det} conf={args.conf}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    t0 = time.time()
    n_done = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        # one predict call per file: list-source streaming in ultralytics
        # 8.4.64 can collapse the whole list into a single giant batch (OOM)
        n_skipped = 0
        for idx, path in enumerate(img_files):
            try:
                res = model.predict(
                    source=path,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    max_det=args.max_det,
                    device=args.device,
                    iou=args.nms_iou,
                    agnostic_nms=args.agnostic_nms,
                    verbose=False,
                )[0]
            except Exception as e:  # corrupt/truncated images (e.g. SKU-110K)
                n_skipped += 1
                print(f"  SKIP {os.path.basename(path)}: {e}", flush=True)
                continue
            boxes = res.boxes
            # xyxy in original image coords; sorted by conf descending
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            order = conf.argsort()[::-1]
            rec = {
                "image": os.path.splitext(os.path.basename(path))[0],
                "width": res.orig_shape[1],
                "height": res.orig_shape[0],
                "boxes": [
                    [round(float(x1), 1), round(float(y1), 1),
                     round(float(x2), 1), round(float(y2), 1),
                     round(float(conf[i]), 4), int(cls[i])]
                    for i in order
                    for x1, y1, x2, y2 in [xyxy[i]]
                ],
            }
            fout.write(json.dumps(rec) + "\n")
            n_done += 1
            if n_done % 100 == 0:
                dt = time.time() - t0
                print(f"  {n_done}/{len(img_files)}  ({dt:.0f}s, "
                      f"{n_done / dt:.1f} img/s)", flush=True)
    print(f"done: {n_done} images ({n_skipped} skipped) -> {args.out}  "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
