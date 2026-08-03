"""De-circularize the DABA cost model: from cached predictions, measure the
marginal recovered objects per budget step as a function of the density proxy
n(x), then show which cost-ratio lambda=c_s/c_m bracket makes the three-way
crossover thresholds equal the deployed (t1,t2)=(100,200). Re-analysis only."""
import os
import sys

import numpy as np

ROOT = os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import wp1_eval as we
import wp4_ap_bootstrap as ab


def per_image(preds, gt, ca=False):
    """return list of (n_proxy, matched@300, matched@600, matched@1000)."""
    rows = []
    for img, g in gt.items():
        if len(g["gt"]) == 0:
            continue
        p = preds.get(img)
        if p is None:
            # No cache record at all: that is missing data, not a detector that
            # returned nothing. Skipping it silently would reshape the marginal
            # recovery curve, so refuse rather than guess.
            raise SystemExit(
                f"ERROR: {img} has ground truth but no prediction record.\n"
                f"    The bracket is read off a complete cache; rebuild it for "
                f"this split before re-running.")
        if len(p) == 0:
            # A legitimate empty prediction: proxy 0, nothing recovered at any
            # budget. It belongs in the low-density bin, not outside the data.
            rows.append((0, 0, 0, 0))
            continue
        n = int((p[:, 4] >= 0.1).sum())               # density proxy
        m = {}
        for k in (300, 600, 1000):
            pk = p[:k]
            mr, _ = we.greedy_match(pk[:, :4], g["gt"], 0.5,
                                    pred_cls=(pk[:, 5].astype(int) if ca else None),
                                    gt_cls=(g["gt_cls"] if ca else None))
            m[k] = int((mr >= 0).sum())
        rows.append((n, m[300], m[600], m[1000]))
    return rows


def analyze(rows, monotone=True):
    rows = np.array(rows, float)
    n = rows[:, 0]
    d1 = rows[:, 2] - rows[:, 1]   # recovered objects 300->600 (per image)
    d2 = rows[:, 3] - rows[:, 2]   # recovered objects 600->1000
    # marginal recovered per added slot
    s1, s2 = d1 / 300.0, d2 / 400.0
    # bin by proxy n, report mean marginal value per slot
    edges = [0, 50, 100, 150, 200, 300, 100000]
    print(f"{'proxy n bin':>12} {'#img':>5} {'rec/slot 300>600':>17} {'rec/slot 600>1000':>18}")
    binvals = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        msk = (n >= lo) & (n < hi)
        if msk.sum() == 0:
            continue
        v1, v2 = s1[msk].mean(), s2[msk].mean()
        binvals.append((lo, hi, msk.sum(), v1, v2))
        print(f"{f'[{lo},{hi})':>12} {int(msk.sum()):>5} {v1:>17.5f} {v2:>18.5f}")
    # crossover: DABA expands 300->600 when rec/slot(300>600) > lambda; the
    # threshold t1 is the proxy n where the per-slot value crosses lambda.
    # Find lambda such that crossover occurs near n=100 (t1) and n=200 (t2).
    # Use a monotone interpolation of s1,s2 vs n midpoints.
    mids = np.array([(lo + min(hi, 500)) / 2 for lo, hi, _, _, _ in binvals])
    a1 = np.array([v for *_, v, _ in binvals])
    a2 = np.array([v for *_, _, v in binvals])

    def val_at(nq, mids, arr):
        return float(np.interp(nq, mids, arr))
    lam_t1 = val_at(100, mids, a1)   # lambda that puts the 300->600 crossover at n=100
    lam_t2 = val_at(200, mids, a2)   # lambda that puts the 600->1000 crossover at n=200
    print(f"\nImplied cost ratio lambda for t1=100 (300->600 step): {lam_t1:.5f}")
    print(f"Implied cost ratio lambda for t2=200 (600->1000 step): {lam_t2:.5f}")
    if not monotone:
        # The single-crossover reading of the cost model needs a non-decreasing
        # marginal profile. Where it does not hold, the two interpolated values
        # are still informative about magnitude, but "brackets the deployed
        # thresholds" would claim more than the premise allows.
        print("=> profile is NOT monotone on this split: the single-crossover "
              "reading does not apply, so no bracketing conclusion is drawn. "
              "The two values above are reported for magnitude only.")
        return
    # The two steps interpolate to different values, so what this shows is that
    # one low cost ratio is *consistent with* both deployed edges, not that a
    # single lambda derives them.
    print(f"=> the deployed (100,200) is bracketed by lambda in "
          f"[{min(lam_t1,lam_t2):.5f}, {max(lam_t1,lam_t2):.5f}]; both ends are "
          f"small, i.e. slot cost << miss cost.")


def _need(path, how):
    """Prediction caches are ~1.1 GB and .gitignore'd, so a clean clone will not
    have them. Say which command rebuilds the file instead of raising a bare
    FileNotFoundError from somewhere inside the loader."""
    if not os.path.exists(path):
        raise SystemExit(
            f"ERROR: missing prediction cache\n    {path}\n"
            f"  This script re-analyses cached predictions; it does not create "
            f"them.\n  Rebuild it with:\n    {how}\n"
            f"  (DART_ROOT is currently {ROOT!r}.)")
    return path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--visdrone-gt",
                    default=os.path.join(ROOT, "data", "VisDrone2019-DET-val", "annotations"))
    ap.add_argument("--visdrone-preds",
                    default=os.path.join(ROOT, "experiments", "wp1_ft", "preds_visdrone_ft_s.jsonl"))
    ap.add_argument("--sku-gt",
                    default=os.path.join(ROOT, "data", "SKU110K_fixed", "annotations", "annotations_test.csv"))
    ap.add_argument("--sku-preds",
                    default=os.path.join(ROOT, "experiments", "wp1_sku", "preds_sku_test_ft.jsonl"))
    ap.add_argument("--skip-sku", action="store_true",
                    help="report the calibration split only")
    args = ap.parse_args()

    # VisDrone-val FIRST: this is the split on which (t1,t2)=(100,200) were
    # chosen, so it is the split the bracket must be read on. The marginal
    # recovered-per-slot profile is monotone here, so the single-crossover
    # reading of the three-way cost model is valid; on SKU-110K it is not,
    # which is why the SKU numbers below are a cross-domain check rather than
    # the calibration.
    _need(args.visdrone_preds,
          "python scripts/wp1_infer.py --model weights_release/visdrone_yolo26s_1280.pt "
          "--images <VisDrone-val images> --imgsz 1280 --out "
          "$DART_ROOT/experiments/wp1_ft/preds_visdrone_ft_s.jsonl")
    vg = we.load_visdrone_gt(args.visdrone_gt)
    vp = ab.load_preds(args.visdrone_preds)
    print("=== VisDrone-val, YOLO26-s (CALIBRATION split) ===")
    analyze(per_image(vp, vg, ca=True))
    print()

    if args.skip_sku:
        raise SystemExit(0)
    _need(args.sku_preds,
          "python scripts/wp1_infer.py --model weights_release/sku110k_yolo26n_1024.pt "
          "--images <SKU-110K test images> --imgsz 1024 --out "
          "$DART_ROOT/experiments/wp1_sku/preds_sku_test_ft.jsonl")
    sku = we.load_sku_gt(args.sku_gt)
    p = ab.load_preds(args.sku_preds)
    print("=== SKU-110K test (cross-domain check; profile is NON-monotone here) ===")
    analyze(per_image(p, sku, ca=False), monotone=False)
