"""DOTAv1 -> tiled HBB detection dataset (2nd aerial domain for the density study).
Steps: tile train/val to 1024 crops (ultralytics split_dota), convert the OBB
crop labels to horizontal YOLO boxes, and report the per-tile object-count
distribution so we can confirm a dense tail exists before finetuning.
"""
import glob
import os
import sys

import numpy as np

ROOT = os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
SRC = os.path.join(DATA, "DOTAv1")          # extracted: images/{train,val}, labels/{train,val}
DST = os.path.join(DATA, "DOTAv1-tiled")    # split_dota output: images/{train,val}, labels/{train,val}


def tile():
    from ultralytics.data.split_dota import split_trainval
    # crop 1024 with 200 overlap, single scale
    split_trainval(data_root=SRC, save_dir=DST, crop_size=1024, gap=200)
    print("tiling done ->", DST)


def obb_to_hbb():
    """Convert split_dota OBB labels (cls x1 y1 x2 y2 x3 y3 x4 y4, normalized)
    to horizontal YOLO labels (cls xc yc w h)."""
    for split in ("train", "val"):
        ld = os.path.join(DST, "labels", split)
        if not os.path.isdir(ld):
            print("missing", ld)
            continue
        n = 0
        for f in glob.glob(os.path.join(ld, "*.txt")):
            out = []
            for line in open(f):
                p = line.split()
                if len(p) < 9:
                    if len(p) == 5:        # already hbb
                        out.append(line.strip())
                    continue
                cls = p[0]
                xs = np.array(p[1:9:2], float)
                ys = np.array(p[2:9:2], float)
                x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
                xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                if w > 0 and h > 0:
                    out.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            open(f, "w").write("\n".join(out) + ("\n" if out else ""))
            n += 1
        print(f"{split}: converted {n} label files to HBB")


def density():
    for split in ("train", "val"):
        ld = os.path.join(DST, "labels", split)
        if not os.path.isdir(ld):
            continue
        cnt = [sum(1 for _ in open(f)) for f in glob.glob(os.path.join(ld, "*.txt"))]
        c = np.array([x for x in cnt if x > 0])
        if len(c) == 0:
            print(split, "no labels")
            continue
        print(f"{split}: {len(c)} non-empty tiles, mean {c.mean():.0f}, median "
              f"{int(np.median(c))}, max {c.max()}, >=150: {(c>=150).sum()}, "
              f">=300: {(c>=300).sum()}, 150-300: {((c>=150)&(c<300)).sum()}")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("all", "tile"):
        tile()
    if step in ("all", "convert"):
        obb_to_hbb()
    if step in ("all", "density"):
        density()
