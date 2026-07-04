"""Convert SKU-110K CSV annotations to YOLO labels + split lists.

CSV rows: image_name,x1,y1,x2,y2,class,image_width,image_height
All images live in SKU110K_fixed/images; we write labels to
SKU110K_fixed/labels (ultralytics derives label path from image path),
plus train.txt/val.txt/test.txt absolute-path lists for the dataset yaml.
Single class 0 (object).
"""
import csv
import os
from collections import defaultdict

ROOT = os.path.join(os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "SKU110K_fixed")


def convert(split):
    ann = os.path.join(ROOT, "annotations", f"annotations_{split}.csv")
    img_dir = os.path.join(ROOT, "images")
    lbl_dir = os.path.join(ROOT, "labels")
    os.makedirs(lbl_dir, exist_ok=True)
    boxes = defaultdict(list)
    skipped = 0
    with open(ann, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 8:
                continue
            name, x1, y1, x2, y2, _, w, h = row[:8]
            x1, y1, x2, y2, w, h = map(float, (x1, y1, x2, y2, w, h))
            if w <= 0 or h <= 0 or x2 <= x1 or y2 <= y1:
                skipped += 1
                continue
            # clamp to image bounds (a few annotations overflow)
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                skipped += 1
                continue
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            boxes[name].append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    n_img, n_box = 0, 0
    listing = []
    for name, lines in boxes.items():
        ip = os.path.join(img_dir, name)
        if not os.path.exists(ip):
            continue
        stem = os.path.splitext(name)[0]
        with open(os.path.join(lbl_dir, stem + ".txt"), "w",
                  encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        listing.append(ip)
        n_img += 1
        n_box += len(lines)
    with open(os.path.join(ROOT, f"{split}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(listing)) + "\n")
    print(f"{split}: {n_img} images, {n_box} boxes, {skipped} skipped")


if __name__ == "__main__":
    for s in ("train", "val", "test"):
        convert(s)
