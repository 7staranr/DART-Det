"""De-circularize the DABA cost model: from cached predictions, measure the
marginal recovered objects per budget step as a function of the density proxy
n(x), then report the span of the two step-specific cost ratios lambda=c_s/c_m
implied by the deployed crossover thresholds (t1,t2)=(100,200).

The two expansion steps yield their own implied cost ratio, so what the output
reports is the span of those two step-specific values -- not a demonstration
that one lambda inside that span derives both thresholds. The single-crossover
reading needs a non-decreasing marginal profile; monotonicity is checked from
the data rather than asserted by the caller. Re-analysis only."""
import os
import sys

import numpy as np

ROOT = os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import wp1_eval as we
import wp4_ap_bootstrap as ab


def per_image(preds, gt, ca=False):
    """return list of (n_proxy, matched@300, matched@600, matched@1000, depth).

    depth is the cached list length, needed to divide by the slots a budget step
    actually adds rather than by its nominal width.

    The analysis is intentionally conditioned on images containing at least
    one ground-truth object. The matched-count increment is perfectly well
    defined for a zero-GT image (it is zero); what such an image cannot carry
    is a recovery rate. Both released splits have none (0 of 548 on
    VisDrone-val, 0 of 2935 on SKU-110K test), so this conditions on nothing
    here; it is stated because on a split that did contain them the reported
    averages would be conditional on having at least one object.
    """
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
            rows.append((0, 0, 0, 0, 0.0))
            continue
        n = int((p[:, 4] >= 0.1).sum())               # density proxy
        m = {}
        for k in (300, 600, 1000):
            pk = p[:k]
            mr, _ = we.greedy_match(pk[:, :4], g["gt"], 0.5,
                                    pred_cls=(pk[:, 5].astype(int) if ca else None),
                                    gt_cls=(g["gt_cls"] if ca else None))
            m[k] = int((mr >= 0).sum())
        rows.append((n, m[300], m[600], m[1000], float(len(p))))
    return rows


def analyze(rows, monotone=None):
    """monotone=None checks the profile from the data; pass a bool only to
    override. An earlier revision took the caller's word for it, so feeding in
    a different cache through the CLI would still have printed the bracketing
    conclusion on a profile that does not support it."""
    rows = np.array(rows, float)
    n = rows[:, 0]
    d1 = rows[:, 2] - rows[:, 1]   # recovered objects 300->600 (per image)
    d2 = rows[:, 3] - rows[:, 2]   # recovered objects 600->1000
    # Per *effective* added slot, not per nominal budget increment: an image
    # whose cache stops at 450 gains 150 returnable slots from a 300->600 step,
    # not 300, and dividing by the nominal width would understate its value.
    # Measured on the released per-image tables, the two agree to five decimals
    # in every proxy bin at or above the deployed thresholds (caches are deep
    # where density is high) and differ only in the sparsest bin, so this does
    # not move the reported crossovers -- it makes the divisor match the label.
    depth = rows[:, 4]
    e1 = np.minimum(depth, 600) - np.minimum(depth, 300)
    e2 = np.minimum(depth, 1000) - np.minimum(depth, 600)
    # An image whose cache adds no slots at a step has no per-slot value to
    # report; clamping its divisor to 1 would call that zero-per-slot and fold
    # it into the mean. Mask it out of that step instead, and say how many.
    ok1, ok2 = e1 > 0, e2 > 0
    s1 = np.divide(d1, e1, out=np.zeros_like(d1), where=ok1)
    s2 = np.divide(d2, e2, out=np.zeros_like(d2), where=ok2)
    # bin by proxy n, report mean marginal value per slot
    edges = [0, 50, 100, 150, 200, 300, 100000]
    print(f"{'proxy n bin':>12} {'#img':>5} {'rec/slot 300>600':>17} {'rec/slot 600>1000':>18}")
    binvals = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        msk = (n >= lo) & (n < hi)
        if msk.sum() == 0:
            continue
        m1, m2 = msk & ok1, msk & ok2
        if m1.sum() == 0 or m2.sum() == 0:
            print(f"{f'[{lo},{hi})':>12} {int(msk.sum()):>5}   "
                  f"no image adds slots at one of the steps; bin skipped")
            continue
        v1, v2 = s1[m1].mean(), s2[m2].mean()
        binvals.append((lo, hi, msk.sum(), v1, v2))
        drop = int(msk.sum() - min(m1.sum(), m2.sum()))
        print(f"{f'[{lo},{hi})':>12} {int(msk.sum()):>5} {v1:>17.5f} {v2:>18.5f}"
              + (f"   ({drop} image(s) add no slots at a step)" if drop else ""))
    # crossover: DABA expands 300->600 when rec/slot(300>600) > lambda; the
    # threshold t1 is the proxy n where the per-slot value crosses lambda.
    # Find lambda such that crossover occurs near n=100 (t1) and n=200 (t2).
    # Use a monotone interpolation of s1,s2 vs n midpoints.
    mids = np.array([(lo + min(hi, 500)) / 2 for lo, hi, _, _, _ in binvals])
    a1 = np.array([v for *_, v, _ in binvals])
    a2 = np.array([v for *_, _, v in binvals])

    if len(binvals) < 2:
        print("insufficient density support: fewer than two usable proxy bins, "
              "so no crossover can be read. No conclusion drawn.")
        return

    def val_at(nq, mids, arr):
        # np.interp clamps outside the observed range instead of failing, which
        # would report an endpoint value as if it were an interpolated
        # crossover. Refuse rather than clamp.
        if not (mids.min() <= nq <= mids.max()):
            raise SystemExit(
                f"ERROR: threshold {nq:g} lies outside the observed density "
                f"range [{mids.min():g}, {mids.max():g}]. Interpolating would "
                f"silently return an endpoint value; this split does not "
                f"support reading a crossover there.")
        return float(np.interp(nq, mids, arr))
    lam_t1 = val_at(100, mids, a1)   # lambda that puts the 300->600 crossover at n=100
    lam_t2 = val_at(200, mids, a2)   # lambda that puts the 600->1000 crossover at n=200
    print(f"\nImplied cost ratio lambda for t1=100 (300->600 step): {lam_t1:.5f}")
    print(f"Implied cost ratio lambda for t2=200 (600->1000 step): {lam_t2:.5f}")
    if monotone is None:
        # Non-decreasing in the density proxy, on both step curves.
        monotone = (all(a1[i] <= a1[i + 1] + 1e-12 for i in range(len(a1) - 1))
                    and all(a2[i] <= a2[i + 1] + 1e-12 for i in range(len(a2) - 1)))
        print(f"marginal profile is "
              f"{'monotone' if monotone else 'NOT monotone'} (checked from the data)")
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
    print(f"=> the two step-specific implied cost ratios span "
          f"[{min(lam_t1,lam_t2):.5f}, {max(lam_t1,lam_t2):.5f}], which is "
          f"consistent with the deployed (100,200); both ends are small, i.e. "
          f"slot cost << miss cost. This does not show that one lambda inside "
          f"that span derives both thresholds.")


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
    # ab.load_preds runs the shared reject_duplicate_keys gate before it
    # collapses the records into a dict, so a duplicated image key fails here
    # too rather than silently last-write-wins.
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
    # No monotone= override: the module promises the check comes from the
    # data, and hard-coding False here would contradict that the moment someone
    # passes a different cache through --sku-preds.
    analyze(per_image(p, sku, ca=False))
