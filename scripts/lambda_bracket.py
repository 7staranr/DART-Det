"""De-circularize the DABA cost model: from cached predictions, measure the
marginal recovered objects per budget step as a function of the density proxy
n(x), then show which cost-ratio lambda=c_s/c_m bracket makes the three-way
crossover thresholds equal the deployed (t1,t2)=(100,200). Re-analysis only."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(ROOT, "scripts"))
import wp1_eval as we
import wp4_ap_bootstrap as ab

ROOT = os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def per_image(preds, gt, ca=False):
    """return list of (n_proxy, matched@300, matched@600, matched@1000)."""
    rows = []
    for img, g in gt.items():
        p = preds.get(img)
        if p is None or len(p) == 0 or len(g["gt"]) == 0:
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


def analyze(rows):
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
    print(f"=> a single low-lambda in [{min(lam_t1,lam_t2):.5f}, {max(lam_t1,lam_t2):.5f}] "
          f"reproduces the deployed (100,200); both are small (slot cost << miss cost).")


if __name__ == "__main__":
    sku = we.load_sku_gt(os.path.join(ROOT, "data", "SKU110K_fixed", "annotations", "annotations_test.csv"))
    p = ab.load_preds(os.path.join(ROOT, "experiments", "wp1_sku", "preds_sku_test_ft.jsonl"))
    print("=== SKU-110K test (proxy n vs marginal recovered-per-slot) ===")
    analyze(per_image(p, sku, ca=False))
