"""Image-bootstrap CI on the dense-subset AP gain (the headline result).

Efficient design: do the expensive greedy matching ONCE per image at each
budget (k=300, k=1000), caching per-image per-class (conf, is_tp) arrays + GT
counts. Then bootstrap resamples IMAGES (2000x) and only re-aggregates AP
(concat -> sort -> cumsum), which is fast. AP@0.5 all-point, class-mean.

Usage: python wp4_ap_bootstrap.py --dataset {sku,visdrone} --preds <jsonl>
  --gt <path> --thr-density 150 [--iou 0.5] [--nboot 2000] [--budgets 300,1000]
"""
import argparse
import json

import numpy as np

import wp1_eval as we


def load_preds(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["image"]] = np.array(r["boxes"], dtype=np.float32).reshape(-1, 6)
    return out


def match_image(p, gt, gcls, k, iou_thr, n_classes):
    """Return per-class (conf[], tp[]) for the top-k ranked preds of one image."""
    res = {}
    if len(p) == 0:
        return res
    p = p[:k]
    boxes, confs, pcls = p[:, :4], p[:, 4], p[:, 5].astype(int)
    for c in range(n_classes):
        cm = pcls == c
        if not cm.any():
            continue
        gboxes = gt[gcls == c]
        pb, pc = boxes[cm], confs[cm]
        order = pc.argsort()[::-1]
        pb, pc = pb[order], pc[order]
        tp = np.zeros(len(pb), dtype=np.int8)
        if len(gboxes):
            ious = we.iou_matrix(pb, gboxes)
            taken = np.zeros(len(gboxes), dtype=bool)
            for r in range(len(pb)):
                row = np.where(~taken, ious[r], -1.0)
                j = int(row.argmax())
                if row[j] >= iou_thr:
                    taken[j] = True
                    tp[r] = 1
        res[c] = (pc, tp)
    return res


def ap_from_cache(samp_idx, cache, ngt_cls, n_classes):
    """cache[i] = {c: (conf, tp)}; aggregate AP over resampled image indices."""
    aps = []
    # accumulate per-class
    confs = {c: [] for c in range(n_classes)}
    tps = {c: [] for c in range(n_classes)}
    ngt = {c: 0 for c in range(n_classes)}
    for i in samp_idx:
        for c, n in ngt_cls[i].items():
            ngt[c] += n
        for c, (pc, tp) in cache[i].items():
            confs[c].append(pc); tps[c].append(tp)
    for c in range(n_classes):
        if ngt[c] == 0:
            continue
        if not confs[c]:
            aps.append(0.0); continue
        cc = np.concatenate(confs[c]); tt = np.concatenate(tps[c])
        order = cc.argsort()[::-1]
        tt = tt[order]
        ctp = np.cumsum(tt); cfp = np.cumsum(1 - tt)
        rec = ctp / ngt[c]
        prec = ctp / np.maximum(ctp + cfp, 1e-9)
        mrec = np.concatenate([[0], rec, [rec[-1]]])
        mpre = np.concatenate([[0], prec, [0]])
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        aps.append(float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])))
    return float(np.mean(aps)) if aps else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["visdrone", "sku", "dota"], required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--thr-density", type=int, default=150)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--budgets", default="300,1000")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    n_classes = 1 if args.dataset == "sku" else (15 if args.dataset == "dota" else 10)
    gt_data = (we.load_sku_gt(args.gt) if args.dataset == "sku"
               else we.load_yolo_gt(args.gt) if args.dataset == "dota"
               else we.load_visdrone_gt(args.gt))
    preds = load_preds(args.preds)
    budgets = [int(x) for x in args.budgets.split(",")]

    dense = [k for k, v in gt_data.items()
             if len(v["gt"]) >= args.thr_density and k in preds]
    print(f"dense images (GT>={args.thr_density}, with preds): {len(dense)}")

    # precompute caches per budget
    caches = {}
    ngt_cls = []
    for img in dense:
        g = gt_data[img]
        ngt_cls.append({c: int((g["gt_cls"] == c).sum())
                        for c in range(n_classes)
                        if int((g["gt_cls"] == c).sum()) > 0})
    for k in budgets:
        caches[k] = [match_image(preds[img], gt_data[img]["gt"],
                                 gt_data[img]["gt_cls"], k, args.iou, n_classes)
                     for img in dense]

    full = np.arange(len(dense))
    pts = {k: ap_from_cache(full, caches[k], ngt_cls, n_classes) for k in budgets}
    klo, khi = budgets[0], budgets[-1]
    print(f"point: AP@{args.iou} " +
          "  ".join(f"max_det={k} {pts[k]:.4f}" for k in budgets) +
          f"  delta({khi}-{klo}) {pts[khi]-pts[klo]:+.4f}")

    rng = np.random.default_rng(args.seed)
    deltas = {k: [] for k in budgets}
    dd = []
    for _ in range(args.nboot):
        samp = rng.integers(0, len(dense), len(dense))
        vals = {k: ap_from_cache(samp, caches[k], ngt_cls, n_classes)
                for k in budgets}
        for k in budgets:
            deltas[k].append(vals[k])
        dd.append(vals[khi] - vals[klo])
    dd = np.array(dd)
    lo, hi = np.percentile(dd, [2.5, 97.5])
    print(f"bootstrap ({args.nboot}x images): AP delta({khi}-{klo}) "
          f"mean {dd.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"P(delta>0)={np.mean(dd > 0):.3f}")
    for k in budgets:
        a = np.array(deltas[k])
        print(f"  AP@{k}: {a.mean():.4f}  CI [{np.percentile(a,2.5):.4f}, "
              f"{np.percentile(a,97.5):.4f}]")


if __name__ == "__main__":
    main()
