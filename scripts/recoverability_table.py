"""Build the RTIL recoverability table: per detector x dataset,
the cache depth M, R@300, R@1000, rel_rec, and DABA-applicability, from the
cached bucket CSVs. Soft top-K heads cache M>K and are recoverable; the hard
query budget caches M==K and is not."""
import csv
import os
ROOT = os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXP = os.path.join(ROOT, "experiments")
ROWS = [
    ("YOLO26-s", "VisDrone", "soft", r"wp1_ft\visdrone_ft_s_buckets.csv"),
    ("YOLO26-n", "SKU-110K", "soft", r"wp1_sku\sku_test_ft_buckets.csv"),
    ("YOLO26-n", "DOTA", "soft", r"wp1_ft\dota_ft_buckets.csv"),
    ("YOLOv10-n", "VisDrone", "soft", r"wp1_ft\visdrone_yolov10n_buckets.csv"),
    ("RT-DETR-L", "VisDrone", "hard", r"wp1_rtdetr\visdrone_rtdetr_buckets.csv"),
    ("RT-DETR-L", "SKU-110K", "hard", r"wp1_sku\sku_rtdetr_buckets.csv"),
]


def getrow(path, bucket):
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["bucket"].strip() == bucket:
                return r
    return None


print(f"{'detector':>10} {'dataset':>9} {'type':>5} {'bucket':>9} {'n':>4} "
      f"{'M':>6} {'R@300':>6} {'R@1000':>7} {'rel_rec':>7} DABA?")
for det, ds, typ, fn in ROWS:
    path = os.path.join(EXP, fn)
    if not os.path.exists(path):
        print(f"{det:>10} {ds:>9}  MISSING {fn}")
        continue
    # report the densest well-populated bucket: >=300 if n>=10 else 150-300
    r300 = getrow(path, ">=300")
    use = r300 if (r300 and int(r300["n_images"]) >= 10) else getrow(path, "150-300")
    if use is None:
        continue
    M = float(use["mean_cache_depth"])
    daba = "yes" if typ == "soft" else "no"
    print(f"{det:>10} {ds:>9} {typ:>5} {use['bucket']:>9} {use['n_images']:>4} "
          f"{M:>6.0f} {float(use['R@300']):>6.3f} {float(use['R@1000']):>7.3f} "
          f"{float(use['rel_rec']):>7.3f} {daba:>5}")
