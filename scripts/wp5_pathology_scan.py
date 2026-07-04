"""WP5 pathology scan: quantify candidate-pool collapse in crowded regions.

Runs the converged ft_n model's o2o assigner (eval mode) on dense + sparse
VisDrone train images and logs per GT:
  - n_inwin   : anchors inside the (STAL-expanded) GT window
  - n_topk    : candidates surviving select_topk (pre-contention zeroing
                counted separately)
  - contested : how many of this GT's topk candidates were zeroed because
                another GT also picked them (tal.py:244 masked_fill)
  - n_final   : positives after full pipeline (0 == DEAD GT: no supervision)
  - crowding  : GT-GT max IoU, neighbor count, image GT count

If dead-GT rate and contention rise steeply with crowding -> the documented
mechanism (candidate nuking) is quantified, motivating the WP5 fix:
contested anchors go to the best GT instead of nobody + dead-GT fallback.
"""
import os

import numpy as np
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    import wp1_eval as we
    from ultralytics import YOLO
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.cfg import get_cfg

    # probe set: reuse WP3's (dense+sparse train images)
    with open(os.path.join(ROOT, "runs", "wp3_dynamics", "probe_ids.txt")) as f:
        probe_ids = sorted(ln.strip() for ln in f if ln.strip())
    gt_all = we.load_visdrone_gt(
        os.path.join(ROOT, "data", "VisDrone2019-DET-train", "annotations"))

    model = YOLO(os.path.join(ROOT, "runs", "ft_visdrone_yolo26n_1280",
                              "weights", "best.pt"))
    dmodel = model.model.cuda(0).eval()

    # loss reads hyp gains from model.args as attribute-style namespace
    cfg0 = get_cfg()
    cfg0.epochs = 1
    dmodel.args = cfg0
    from ultralytics.utils.loss import E2ELoss
    crit = E2ELoss(dmodel)
    assigner = crit.one2one.assigner

    # stash internals via monkey-patch
    stash = {}
    orig_topk = assigner.select_topk_candidates

    def patched_topk(metrics, topk_mask=None):
        topk_metrics, topk_idxs = torch.topk(metrics, assigner.topk, dim=-1,
                                             largest=True)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] >
                         assigner.eps).expand_as(topk_idxs)
        topk_idxs = topk_idxs.masked_fill(~topk_mask, 0)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8,
                                   device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8)
        for k in range(assigner.topk):
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k:k + 1], ones)
        contested_map = count_tensor > 1            # (b, n_gt, n_anchor)
        out = count_tensor.masked_fill(contested_map, 0).to(metrics.dtype)
        # per GT: how many of its picks were nuked by contention
        stash["picked"] = (count_tensor >= 1)
        stash["contested"] = contested_map
        return out

    assigner.select_topk_candidates = patched_topk

    orig_fwd = assigner.forward

    def patched_fwd(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes,
                    mask_gt):
        out = orig_fwd(pd_scores, pd_bboxes, anc_points, gt_labels,
                       gt_bboxes, mask_gt)
        mask_in = assigner.select_candidates_in_gts(anc_points, gt_bboxes,
                                                    mask_gt)
        fg_mask, target_gt_idx = out[3], out[4]
        b_sz, n_gt_max = mask_gt.shape[0], mask_gt.shape[1]
        n_final = torch.zeros(b_sz, n_gt_max, device=fg_mask.device)
        bidx, aidx = torch.nonzero(fg_mask, as_tuple=True)
        n_final.index_put_((bidx, target_gt_idx[bidx, aidx]),
                           torch.ones_like(bidx, dtype=n_final.dtype),
                           accumulate=True)
        stash["n_inwin"] = mask_in.sum(-1).cpu().numpy()
        stash["n_contested"] = (stash.pop("contested") &
                                stash.pop("picked")).sum(-1).cpu().numpy()
        stash["n_final"] = n_final.cpu().numpy()
        stash["mask_gt"] = mask_gt.squeeze(-1).cpu().numpy()
        return out

    assigner.forward = patched_fwd

    cfg = get_cfg()
    cfg.imgsz = 1280
    ds = build_yolo_dataset(
        cfg, os.path.join(ROOT, "data", "VisDrone2019-DET-train", "images"),
        batch=4, data={"names": {i: str(i) for i in range(10)},
                       "channels": 3},
        mode="val", rect=False, stride=32)
    keep = [i for i, f in enumerate(ds.im_files)
            if os.path.splitext(os.path.basename(f))[0] in set(probe_ids)]
    ds.im_files = [ds.im_files[i] for i in keep]
    ds.labels = [ds.labels[i] for i in keep]
    loader = build_dataloader(ds, batch=4, workers=0, shuffle=False, rank=-1)

    # covariates in the same (batch, slot, gt) order
    recs = []
    img_pos = 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch["img"].cuda(0).float() / 255
            preds = dmodel(imgs)
            batch_gpu = {k: (v.cuda(0) if isinstance(v, torch.Tensor) else v)
                         for k, v in batch.items()}
            try:
                crit(preds, batch_gpu)
            except Exception as e:
                print("loss err:", e)
                break
            mgt = stash.pop("mask_gt")
            n_inwin = stash.pop("n_inwin")
            n_cont = stash.pop("n_contested")
            n_fin = stash.pop("n_final")
            for b in range(mgt.shape[0]):
                img_id = probe_ids[img_pos + b]
                gt = gt_all[img_id]["gt"]
                ious = we.iou_matrix(gt, gt)
                np.fill_diagonal(ious, 0)
                mi = ious.max(1) if len(gt) > 1 else np.zeros(len(gt))
                n = int(mgt[b].sum())
                for gi in range(min(n, len(gt))):
                    recs.append((len(gt), float(mi[gi]),
                                 float(n_inwin[b, gi]), float(n_cont[b, gi]),
                                 float(n_fin[b, gi])))
            img_pos += mgt.shape[0]
    arr = np.array(recs)
    print(f"GT records: {len(arr)}")

    def strat(name, sel):
        s = arr[sel]
        if len(s) < 100:
            return
        dead = (s[:, 4] == 0).mean()
        print(f"  {name:>18}: n_inwin={s[:, 2].mean():6.1f}  "
              f"contested={s[:, 3].mean():5.2f}  "
              f"DEAD-GT rate={dead:6.3f}  n={len(s)}")

    print("\n--- by local crowding (GT-GT max IoU) ---")
    strat("isolated <0.1", arr[:, 1] < 0.1)
    strat("touch 0.1-0.3", (arr[:, 1] >= 0.1) & (arr[:, 1] < 0.3))
    strat("overlap 0.3-0.6", (arr[:, 1] >= 0.3) & (arr[:, 1] < 0.6))
    strat("heavy >0.6", arr[:, 1] >= 0.6)
    print("--- by image density ---")
    strat("sparse <50", arr[:, 0] < 50)
    strat("dense >=150", arr[:, 0] >= 150)
    print("--- dense x crowding ---")
    strat("dense/isolated", (arr[:, 0] >= 150) & (arr[:, 1] < 0.1))
    strat("dense/crowded>.3", (arr[:, 0] >= 150) & (arr[:, 1] >= 0.3))

    out = os.path.join(ROOT, "experiments", "wp5", "pathology_scan.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, recs=arr)
    print("wrote", out)


if __name__ == "__main__":
    main()
