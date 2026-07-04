"""WP3 offline analysis: assignment flip rate & stability margin vs crowding.

Reads runs/wp3_dynamics/probe_epoch{N}.npz (rows: epoch, batch_i, b, gt_i,
sel_anchor, margin). GT identity is stable across epochs because the probe
loader is fixed (shuffle=False, rect=False, same order every epoch).

Joins with VisDrone train GT to attach crowding covariates (GT-GT max IoU,
neighbor count, image GT count), then reports:
  - flip rate (selected anchor changes between consecutive epochs) stratified
    by local crowding x image density
  - mean stability margin by the same strata
Training-time H2 predicts: crowded GTs flip more and have smaller margins.
"""
import os
import re
from collections import defaultdict

import sys

import numpy as np

import wp1_eval as we

ROOT = os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# DYN dir holds probe_epoch*.npz + probe_ids.txt; override via argv[1].
DYN = (sys.argv[1] if len(sys.argv) > 1
       else os.path.join(ROOT, "runs", "wp3_dynamics"))


def load_epochs():
    files = {}
    for fn in os.listdir(DYN):
        m = re.match(r"probe_epoch(\d+)\.npz", fn)
        if m:
            files[int(m.group(1))] = os.path.join(DYN, fn)
    out = {}
    for ep in sorted(files):
        rows = np.load(files[ep])["rows"]
        # key = (batch_i, b, gt_i) -> (sel_anchor, margin)
        d = {(int(r[1]), int(r[2]), int(r[3])): (int(r[4]), float(r[5]))
             for r in rows}
        out[ep] = d
    return out


def crowding_covariates():
    """Covariates per probe GT key. Requires probe loader order replication:
    probe_ids.txt order, batches of 4, GT order as in labels (mask_gt order
    = label order)."""
    with open(os.path.join(DYN, "probe_ids.txt")) as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    # NOTE: the val-mode dataset sorts im_files; replicate sorted order
    gt_all = we.load_visdrone_gt(
        os.path.join(ROOT, "data", "VisDrone2019-DET-train", "annotations"))
    id_sorted = sorted(ids)
    cov = {}
    for pos, img in enumerate(id_sorted):
        gt = gt_all[img]["gt"]
        n_gt = len(gt)
        ious = we.iou_matrix(gt, gt)
        np.fill_diagonal(ious, 0)
        max_iou = ious.max(1) if n_gt > 1 else np.zeros(n_gt)
        ctr = np.stack([(gt[:, 0] + gt[:, 2]) / 2,
                        (gt[:, 1] + gt[:, 3]) / 2], 1)
        area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
        rad = 2 * np.sqrt(np.maximum(area, 1))
        d2 = ((ctr[:, None] - ctr[None]) ** 2).sum(-1)
        nbr = (d2 <= rad[:, None] ** 2).sum(1) - 1
        bi, b = divmod(pos, 4)  # loader batch=4, shuffle=False
        for gi in range(n_gt):
            cov[(bi, b, gi)] = (float(max_iou[gi]), int(nbr[gi]), n_gt)
    return cov


def main():
    eps = load_epochs()
    cov = crowding_covariates()
    epochs = sorted(eps)
    print(f"epochs: {epochs[0]}..{epochs[-1]}  probe GTs: {len(eps[epochs[0]])}")

    # flip events between consecutive epochs
    recs = []
    for e0, e1 in zip(epochs[:-1], epochs[1:]):
        d0, d1 = eps[e0], eps[e1]
        for key in d0.keys() & d1.keys():
            if key not in cov:
                continue
            a0, m0 = d0[key]
            a1, _ = d1[key]
            if a0 < 0 or a1 < 0:
                continue
            mi, nbr, ngt = cov[key]
            recs.append((int(a0 != a1), m0, mi, nbr, ngt, e1))
    arr = np.array(recs)
    print(f"transition records: {len(arr)}")

    def strat(name, sel):
        s = arr[sel]
        if len(s) < 200:
            print(f"  {name:>24}: n={len(s)} (too few)")
            return
        late = s[s[:, 5] > epochs[len(epochs) // 2]]
        print(f"  {name:>24}: flip={s[:, 0].mean():.3f} "
              f"(late-half {late[:, 0].mean():.3f})  "
              f"margin={s[:, 1].mean():.4f}  n={len(s)}")

    print("\n--- by local crowding (GT-GT max IoU) ---")
    strat("isolated <0.1", arr[:, 2] < 0.1)
    strat("touch 0.1-0.3", (arr[:, 2] >= 0.1) & (arr[:, 2] < 0.3))
    strat("overlap 0.3-0.6", (arr[:, 2] >= 0.3) & (arr[:, 2] < 0.6))
    strat("heavy >0.6", arr[:, 2] >= 0.6)
    print("\n--- by neighbor count ---")
    strat("0 nbrs", arr[:, 3] == 0)
    strat("1-3 nbrs", (arr[:, 3] >= 1) & (arr[:, 3] < 4))
    strat("4-9 nbrs", (arr[:, 3] >= 4) & (arr[:, 3] < 10))
    strat(">=10 nbrs", arr[:, 3] >= 10)
    print("\n--- by image density ---")
    strat("sparse <50", arr[:, 4] < 50)
    strat("dense >=150", arr[:, 4] >= 150)
    print("\n--- crowding x density (flip rate) ---")
    for dname, dsel in (("sparse", arr[:, 4] < 50), ("dense", arr[:, 4] >= 150)):
        for cname, csel in (("isolated", arr[:, 2] < 0.1),
                            ("crowded>0.3", arr[:, 2] >= 0.3)):
            strat(f"{dname}/{cname}", dsel & csel)


if __name__ == "__main__":
    main()
