"""Build the RTIL recoverability table: per detector x dataset,
the cache depth M, R@300, R@1000, rel_rec, and DABA-applicability, from the
cached bucket CSVs. The inference-time recoverability condition is about the
cache, not the family label. Retaining an accessible rank tail (observed depth
> K) is what makes a head structurally eligible for the repair; whether the
repair actually gains on a given split is the separate, empirical question this
table answers with R@1000 > R@300. Soft top-K heads are eligible; the evaluated
query budget caches M==K and is not."""
import csv
import io
import os
ROOT = os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# results/ is what the release ships; experiments/ is the authors' working tree
# (gitignored). Try the shipped location first so this runs on a clean clone.
_ROOTS = [os.path.join(ROOT, "results", "buckets"), os.path.join(ROOT, "experiments")]


def _find(rel):
    rel = rel.replace("\\", os.sep).replace("/", os.sep)
    for base in _ROOTS:
        p = os.path.join(base, rel)
        if os.path.exists(p):
            return p
    return os.path.join(_ROOTS[0], rel)
ROWS = [
    ("YOLO26-s", "VisDrone", "soft", r"wp1_ft\visdrone_ft_s_buckets.csv"),
    ("YOLO26-n", "SKU-110K", "soft", r"wp1_sku\sku_test_ft_buckets.csv"),
    ("YOLO26-n", "DOTA", "soft", r"wp1_ft\dota_ft_buckets.csv"),
    ("YOLOv10-n", "VisDrone", "soft", r"wp1_ft\visdrone_yolov10n_buckets.csv"),
    ("RT-DETR-L", "VisDrone", "hard", r"wp1_rtdetr\visdrone_rtdetr_buckets.csv"),
    ("RT-DETR-L", "SKU-110K", "hard", r"wp1_sku\sku_rtdetr_buckets.csv"),
]


def tail_evidence(bucket_rel, bucket, k=300):
    """Fraction of that bucket's images whose cache actually holds a tail.

    Uses results/per_image/, which the release ships. Returns (frac, max_depth)
    or (None, None) when the per-image table is unavailable, in which case the
    caller falls back to the bucket mean and says so.
    """
    rel = bucket_rel.replace("_buckets.csv", "_per_image.csv")
    rel = rel.replace("\\", os.sep).replace("/", os.sep)
    for base in (os.path.join(ROOT, "results", "per_image"),
                 os.path.join(ROOT, "experiments")):
        path = os.path.join(base, rel)
        if os.path.exists(path):
            break
    else:
        return None, None
    depths = [float(r["cache_depth"]) for r in
              csv.DictReader(io.open(path, encoding="utf-8"))
              if r["bucket"].strip() == bucket]
    if not depths:
        return None, None
    return sum(d > k for d in depths) / len(depths), max(depths)


def getrow(path, bucket):
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["bucket"].strip() == bucket:
                return r
    return None


# "obs.M" is the observed mean number of retained predictions in the bucket,
# not the configured cache cap: it is bounded by the cap but also by the
# confidence floor and by how many candidates the image actually yields.
bases = []
print(f"{'detector':>10} {'dataset':>9} {'type':>5} {'bucket':>9} {'n':>4} "
      f"{'obs.M':>6} {'R@300':>6} {'R@1000':>7} {'rel_rec':>7} "
      f"{'tail?':>6} {'gain?':>6} {'DABA?':>6}")
for det, ds, typ, fn in ROWS:
    path = _find(fn)
    if not os.path.exists(path):
        print(f"{det:>10} {ds:>9}  MISSING {fn}")
        continue
    # report the densest well-populated bucket: >=300 if n>=10 else 150-300
    r300 = getrow(path, ">=300")
    use = r300 if (r300 and int(r300["n_images"]) >= 10) else getrow(path, "150-300")
    if use is None:
        continue
    M = float(use["mean_cache_depth"])
    # Decide from the measured cache, not from the family label: DABA can act
    # only where an accessible rank tail exists beyond the deployed budget.
    # mean_cache_depth is an observed mean, not the configured cap, so it is
    # reported as such; rel_rec == 0 means nothing below rank K to reopen.
    # Structural: does a reopenable tail exist? Decided per image, not from
    # the bucket mean -- a mean can sit below K while plenty of images hold a
    # tail, which would print tail?=no next to gain?=yes.
    frac, max_depth = tail_evidence(fn, use["bucket"])
    if frac is None:
        eligible, basis = M > 300.0 + 1e-9, "bucket mean (per-image table absent)"
    else:
        eligible, basis = frac > 0.0, f"{frac:.0%} of images, max depth {max_depth:.0f}"
    gained = float(use["rel_rec"]) > 0.0  # empirical: it pays off on this split
    e_s = "yes" if eligible else "no"
    g_s = "yes" if gained else "no"
    daba = "yes" if (eligible and gained) else "no"
    bases.append((det, ds, basis))
    print(f"{det:>10} {ds:>9} {typ:>5} {use['bucket']:>9} {use['n_images']:>4} "
          f"{M:>6.0f} {float(use['R@300']):>6.3f} {float(use['R@1000']):>7.3f} "
          f"{float(use['rel_rec']):>7.3f} {e_s:>6} {g_s:>6} {daba:>6}")

print("\ntail? basis (fraction of the bucket's images whose cache_depth > 300):")
for det, ds, basis in bases:
    print(f"  {det:>10} {ds:>9}  {basis}")
