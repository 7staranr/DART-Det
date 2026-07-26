# -*- coding: utf-8 -*-
"""Robustness of the three-way decomposition to the choice of dense endpoint.

The published split uses <50 -> >=300 on VisDrone-val, whose >=300 bucket holds
only 3 images and is declared qualitative-only by the protocol. This recomputes
the same decomposition with the 150-300 bucket (n=27) as the dense endpoint.

    total decay      = R@300(sparse)  - R@300(dense)          [end-to-end]
    budget truncation= total decay    - (R@1000 decay)        [recoverable]
    shared difficulty= R@1000 decay of the one-to-many+NMS reference path
    head residual    = (e2e R@1000 decay) - (o2m R@1000 decay)
"""
import csv, os, sys
sys.stdout.reconfigure(encoding='utf-8')
E = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "experiments", "wp1_ft")


def load(f):
    p = os.path.join(E, f)
    if not os.path.exists(p):
        return None
    return {r["bucket"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}


def decomp(e2e, o2m, sparse, dense, tag):
    if not e2e or not o2m or dense not in e2e or dense not in o2m:
        print(f"  {tag}: data missing"); return
    d300 = float(e2e[sparse]["R@300"]) - float(e2e[dense]["R@300"])
    d1000 = float(e2e[sparse]["R@1000"]) - float(e2e[dense]["R@1000"])
    shared = float(o2m[sparse]["R@1000"]) - float(o2m[dense]["R@1000"])
    budget = d300 - d1000
    head = d1000 - shared
    tot = d300
    print(f"  {tag}")
    print(f"     total R@300 decay          {tot:+.4f}")
    print(f"     budget truncation          {budget:+.4f}  ({budget/tot*100:4.1f}%)")
    print(f"     shared dense difficulty    {shared:+.4f}  ({shared/tot*100:4.1f}%)")
    print(f"     one-to-one head residual   {head:+.4f}  ({head/tot*100:4.1f}%)")
    print(f"     (sum {budget+shared+head:+.4f})")


for model, e2ef in [("YOLO26-s", "visdrone_ft_s_buckets.csv"),
                    ("YOLO26-n", "visdrone_ft_n_buckets.csv")]:
    e2e = load(e2ef)
    for o2mf in ["ft_s_o2m_nms05_ag_buckets.csv", "ft_s_o2m_nms05_buckets.csv"]:
        o2m = load(o2mf)
        if not o2m:
            continue
        print(f"\n=== {model}   reference: {o2mf} ===")
        decomp(e2e, o2m, "<50", ">=300", "dense endpoint >=300  (n=3, PUBLISHED)")
        decomp(e2e, o2m, "<50", "150-300", "dense endpoint 150-300 (n=27, admissible)")
    break  # o2m reference was produced for the -s weights only
