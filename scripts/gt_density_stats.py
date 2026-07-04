"""GT density statistics for VisDrone-val and CrowdHuman-val.

Buckets images by ground-truth object count and reports the distribution.
This determines whether each benchmark has enough high-density images to
support the density-stratified recall analysis.

VisDrone annotation format (one txt per image, one object per line):
    bbox_left,bbox_top,bbox_width,bbox_height,score,object_category,truncation,occlusion
    category 0 = ignored region, 11 = others -> excluded from GT count
    categories 1-10 are valid evaluation classes.

CrowdHuman annotation format: annotation_val.odgt, one JSON per line:
    {"ID": ..., "gtboxes": [{"tag": "person"|"mask", "fbox": [x,y,w,h],
                             "extra": {"ignore": 0|1}, ...}, ...]}
    tag != "person" or extra.ignore == 1 -> excluded.
"""
import json
import os
ROOT = os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
from collections import Counter

BUCKETS = [(0, 50), (50, 100), (100, 150), (150, 300), (300, 10**9)]
BUCKET_NAMES = ["<50", "50-100", "100-150", "150-300", ">300"]


def bucket_of(n):
    for (lo, hi), name in zip(BUCKETS, BUCKET_NAMES):
        if lo <= n < hi:
            return name
    return BUCKET_NAMES[-1]


def visdrone_counts(ann_dir):
    counts = {}
    for fn in os.listdir(ann_dir):
        if not fn.endswith(".txt"):
            continue
        n = 0
        with open(os.path.join(ann_dir, fn), "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                cat = int(parts[5])
                if 1 <= cat <= 10:
                    n += 1
        counts[fn[:-4]] = n
    return counts


def crowdhuman_counts(odgt_path):
    counts = {}
    with open(odgt_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            n = 0
            for gb in rec.get("gtboxes", []):
                if gb.get("tag") != "person":
                    continue
                if gb.get("extra", {}).get("ignore", 0) == 1:
                    continue
                n += 1
            counts[rec["ID"]] = n
    return counts


def report(name, counts):
    vals = sorted(counts.values())
    n = len(vals)
    total = sum(vals)
    mean = total / n
    median = vals[n // 2]
    p90 = vals[int(n * 0.9)]
    p99 = vals[int(n * 0.99)]
    mx = vals[-1]
    print(f"\n=== {name} ===")
    print(f"images={n}  total_gt={total}  mean={mean:.1f}  median={median}  "
          f"p90={p90}  p99={p99}  max={mx}")
    dist = Counter(bucket_of(v) for v in vals)
    print("bucket distribution:")
    for bname in BUCKET_NAMES:
        c = dist.get(bname, 0)
        print(f"  {bname:>8}: {c:5d} images ({100.0 * c / n:5.1f}%)")
    over = sum(1 for v in vals if v >= 300)
    near = sum(1 for v in vals if 150 <= v < 300)
    print(f"images with GT>=300 (budget mathematically insufficient): {over}")
    print(f"images with 150<=GT<300 (budget pressure zone):          {near}")


if __name__ == "__main__":
    base = os.path.join(ROOT, "data")
    vd_ann = os.path.join(base, "VisDrone2019-DET-val", "annotations")
    if os.path.isdir(vd_ann):
        report("VisDrone2019-DET-val", visdrone_counts(vd_ann))
    ch_odgt = os.path.join(base, "crowdhuman", "annotation_val.odgt")
    if os.path.isfile(ch_odgt):
        report("CrowdHuman-val", crowdhuman_counts(ch_odgt))
    else:
        print("\nCrowdHuman odgt not present yet, skipped.")
