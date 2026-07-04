"""Convert VisDrone-DET annotations to YOLO format (ultralytics convention).

VisDrone categories 1-10 -> YOLO classes 0-9:
  0 pedestrian, 1 people, 2 bicycle, 3 car, 4 van, 5 truck,
  6 tricycle, 7 awning-tricycle, 8 bus, 9 motor
Category 0 (ignored regions) and 11 (others) are skipped.
Matches ultralytics' official VisDrone.yaml conversion logic.

Usage:
  python visdrone_to_yolo.py --root /path/to/VisDrone2019-DET-train
Creates <root>/labels/*.txt alongside <root>/images.
"""
import argparse
import os

from PIL import Image


def convert_split(root):
    ann_dir = os.path.join(root, "annotations")
    img_dir = os.path.join(root, "images")
    lbl_dir = os.path.join(root, "labels")
    os.makedirs(lbl_dir, exist_ok=True)
    n_img, n_box = 0, 0
    for fn in sorted(os.listdir(ann_dir)):
        if not fn.endswith(".txt"):
            continue
        stem = fn[:-4]
        img_path = os.path.join(img_dir, stem + ".jpg")
        if not os.path.exists(img_path):
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        lines = []
        with open(os.path.join(ann_dir, fn), "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip().split(",")
                if len(p) < 6:
                    continue
                cat = int(p[5])
                if not (1 <= cat <= 10):
                    continue
                bx, by, bw, bh = float(p[0]), float(p[1]), float(p[2]), float(p[3])
                if bw <= 0 or bh <= 0:
                    continue
                cx = (bx + bw / 2) / w
                cy = (by + bh / 2) / h
                lines.append(f"{cat - 1} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
        with open(os.path.join(lbl_dir, stem + ".txt"), "w",
                  encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        n_img += 1
        n_box += len(lines)
    print(f"{root}: {n_img} images, {n_box} boxes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True)
    args = ap.parse_args()
    for r in args.root:
        convert_split(r)
