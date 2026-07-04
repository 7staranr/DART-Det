"""Paper-level statistics: image-clustered logistic regression of
per-GT detection@300 on density, controlling for object size and local
crowding. Gives the density effect a coefficient + image-cluster-robust CI —
the rigorous version of the recall-density decline, pooled over seeds.

Model: detect@300 ~ log(image_gt_count) + log(area) + local_max_iou + nbr_count
Cluster-robust SE by image (each image is a cluster of correlated GTs).
Also a mixed-effects logit (image random intercept) if it converges.

Usage: python diag_glmm.py --per-gt <csv> [<csv2> ...]  (pools seeds/datasets)
"""
import argparse

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-gt", nargs="+", required=True)
    ap.add_argument("--label", default="diagnosis")
    args = ap.parse_args()

    frames = []
    for si, p in enumerate(args.per_gt):
        df = pd.read_csv(p)
        df["seed_img"] = f"s{si}_" + df["image"].astype(str)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["n_gt"] > 0].copy()
    df["log_ngt"] = np.log(df["n_gt"])
    df["log_area"] = np.log(np.clip(df["area"], 1, None))
    df["crowd"] = df["max_iou_nbr"]
    df["nbr"] = df["nbr_count"]
    df["y"] = df["m300"].astype(int)
    print(f"{args.label}: {len(df)} GTs, {df['seed_img'].nunique()} image-"
          f"clusters, detect@300 rate {df['y'].mean():.3f}")

    # cluster-robust logit
    m = smf.logit("y ~ log_ngt + log_area + crowd + nbr", data=df).fit(
        disp=False, cov_type="cluster",
        cov_kwds={"groups": df["seed_img"]})
    print("\n=== image-cluster-robust logistic regression ===")
    print(f"{'term':>12} {'coef':>9} {'OR':>8} {'z':>8} {'p':>10} "
          f"{'95% CI (coef)':>22}")
    ci = m.conf_int()
    for term in m.params.index:
        b = m.params[term]
        z = m.tvalues[term]
        p = m.pvalues[term]
        lo, hi = ci.loc[term]
        print(f"{term:>12} {b:>9.3f} {np.exp(b):>8.3f} {z:>8.2f} {p:>10.2e} "
              f"[{lo:>7.3f},{hi:>7.3f}]")
    print("\n(log_ngt coef < 0 with CI excluding 0 = density independently "
          "lowers detection prob, controlling for object size & local "
          "crowding. OR = odds multiplier per e-fold density increase.)")

    # marginal effect: predicted detect@300 across density holding others at mean
    print("\n=== predicted detect@300 vs image GT count (covariates at mean) ===")
    base = {c: df[c].mean() for c in ["log_area", "crowd", "nbr"]}
    for ngt in (20, 50, 100, 200, 400):
        row = pd.DataFrame([{**base, "log_ngt": np.log(ngt)}])
        pr = m.predict(row).iloc[0]
        print(f"  GT/img={ngt:>4}: P(detect@300)={pr:.3f}")


if __name__ == "__main__":
    main()
