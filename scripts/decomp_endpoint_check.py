# -*- coding: utf-8 -*-
"""Robustness of the three-way decomposition to the choice of dense endpoint.

The primary endpoint is <50 -> 150-300 on VisDrone-val (n=27). The >=300
bucket holds only 3 images and is declared qualitative-only by the protocol, so
it is reported as a sensitivity check, not as the headline split. Both are
printed below, primary first.

    total decay      = R@300(sparse)  - R@300(dense)          [end-to-end]
    budget truncation= total decay    - (R@1000 decay)        [recoverable]
    shared difficulty= R@1000 decay of the one-to-many+NMS reference path
    path residual    = (e2e R@1000 decay) - (o2m R@1000 decay)

The path residual is the net difference between the two evaluated decode paths,
which differ in head, supervision, assignment, calibration, NMS and matching
convention. It is not an isolated one-to-one-head effect: those factors may
partly cancel, so its size does not bound the head's own contribution.
"""
import csv, os, sys
sys.stdout.reconfigure(encoding='utf-8')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# results/buckets/wp1_ft is what the release ships; experiments/wp1_ft is the
# authors' working tree (excluded by .gitignore). Try the shipped path first so
# this runs against a clean clone.
SEARCH = [os.path.join(ROOT, "results", "buckets", "wp1_ft"),
          os.path.join(ROOT, "experiments", "wp1_ft")]


def load(f):
    for d in SEARCH:
        p = os.path.join(d, f)
        if os.path.exists(p):
            break
    else:
        return None
    return {r["bucket"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}


def decomp(e2e, o2m, sparse, dense, tag):
    if not e2e or not o2m or dense not in e2e or dense not in o2m:
        print(f"  {tag}: data missing"); return
    d300 = float(e2e[sparse]["R@300"]) - float(e2e[dense]["R@300"])
    d1000 = float(e2e[sparse]["R@1000"]) - float(e2e[dense]["R@1000"])
    shared = float(o2m[sparse]["R@1000"]) - float(o2m[dense]["R@1000"])
    budget = d300 - d1000
    path_residual = d1000 - shared
    tot = d300
    if abs(tot) < 1e-12:
        print(f"  {tag}: no sparse-to-dense decay to decompose"); return
    print(f"  {tag}")
    print(f"     total R@300 decay          {tot:+.4f}")
    print(f"     budget truncation          {budget:+.4f}  ({budget/tot*100:4.1f}%)")
    print(f"     shared dense difficulty    {shared:+.4f}  ({shared/tot*100:4.1f}%)")
    print(f"     path residual (o2o vs o2m) {path_residual:+.4f}  ({path_residual/tot*100:4.1f}%)")
    print(f"     (sum {budget+shared+path_residual:+.4f})")


# The one-to-many+NMS reference must be the same one Table 1 is built from --
# the plain per-scale o2m cache. The class-agnostic NMS variants below are a
# different matching convention and give a different (larger) shared term, so
# mixing them in silently makes the decomposition look endpoint-sensitive when
# it is not.
for model, e2ef, o2mf in [("YOLO26-s", "visdrone_ft_s_buckets.csv",
                                       "visdrone_ft_s_o2m_buckets.csv"),
                          ("YOLO26-n", "visdrone_ft_n_buckets.csv",
                                       "visdrone_ft_n_o2m_buckets.csv")]:
    e2e, o2m = load(e2ef), load(o2mf)
    if not e2e or not o2m:
        continue
    print(f"\n=== {model}   reference: {o2mf} ===")
    decomp(e2e, o2m, "<50", "150-300", "dense endpoint 150-300 (n=27, PRIMARY)")
    decomp(e2e, o2m, "<50", ">=300", "dense endpoint >=300   (n=3, SENSITIVITY ONLY)")

# For completeness: the same check against the class-agnostic NMS references,
# which is what an earlier revision of this script used by mistake.
print("\n--- sensitivity to the NMS convention of the reference path ---")
for o2mf in ["ft_s_o2m_nms05_ag_buckets.csv", "ft_s_o2m_nms05_buckets.csv"]:
    o2m = load(o2mf)
    if not o2m:
        continue
    print(f"\n=== YOLO26-s   reference: {o2mf} ===")
    decomp(load("visdrone_ft_s_buckets.csv"), o2m, "<50", "150-300",
           "dense endpoint 150-300 (primary)")
    decomp(load("visdrone_ft_s_buckets.csv"), o2m, "<50", ">=300",
           "dense endpoint >=300   (sensitivity)")
