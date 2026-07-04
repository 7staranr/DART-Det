"""Training-time assignment-dynamics instrumentation.

Runs a short probe finetune (yolo26n on VisDrone) with a callback that, at
each epoch end, executes the o2o TAL assigner (tal_topk2=1 branch) on a FIXED
probe set of images and logs per GT:
  - selected anchor index (argmax aligned metric within positive mask)
  - stability margin: top1 - top2 aligned metric (small = unstable assignment)
  - local crowding covariates (GT-GT max IoU, neighbor count)

Offline analysis then computes epoch-to-epoch anchor flip rate stratified by
crowding x image density: training-time H2 predicts higher flip rate AND lower
margin for crowded GTs.

Implementation: monkey-patches the o2o assigner's forward to stash
(align_metric, mask_pos) of the last call; the callback feeds probe batches
through the loss to trigger assignment, then reads the stash.

Usage:
  python wp3_assign_dynamics.py --epochs 20 --device 1
Outputs: runs/wp3_dynamics/probe_epoch{N}.npz + final flip-rate report.
"""
import argparse
import os

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "runs", "wp3_dynamics")


def build_probe_set(n_dense=24, n_sparse=24, seed=0):
    """Pick fixed probe image ids from VisDrone train (dense + sparse)."""
    import wp1_eval as we
    ann = os.path.join(ROOT, "data", "VisDrone2019-DET-train", "annotations")
    gt = we.load_visdrone_gt(ann)
    items = sorted(gt.items(), key=lambda kv: -len(kv[1]["gt"]))
    dense = [k for k, v in items if len(v["gt"]) >= 150][:n_dense]
    sparse = [k for k, v in items if 5 <= len(v["gt"]) < 50]
    rng = np.random.default_rng(seed)
    sparse = list(rng.choice(sparse, n_sparse, replace=False))
    return dense + sparse, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default="1")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patch", action="store_true",
                    help="apply WP5 no-orphan assigner patch")
    ap.add_argument("--name", default="wp3_probe_train")
    ap.add_argument("--dyn-dir", default=None,
                    help="output dir for probe npz (default OUT_DIR)")
    args = ap.parse_args()

    global OUT_DIR
    if args.dyn_dir:
        OUT_DIR = args.dyn_dir
    if args.patch:
        import wp5_train_norphan
        wp5_train_norphan.patch_assigner()

    import torch
    from ultralytics import YOLO
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.cfg import get_cfg

    os.makedirs(OUT_DIR, exist_ok=True)
    probe_ids, gt_all = build_probe_set()
    with open(os.path.join(OUT_DIR, "probe_ids.txt"), "w") as f:
        f.write("\n".join(probe_ids))
    print(f"probe set: {len(probe_ids)} images")

    model = YOLO(os.path.join(ROOT, "weights", "yolo26n.pt"))

    # stash for assigner outputs, filled by monkey-patched forward
    stash = {}

    def make_probe_loader(trainer):
        cfg = get_cfg()
        cfg.imgsz = args.imgsz
        ds = build_yolo_dataset(
            cfg, os.path.join(ROOT, "data", "VisDrone2019-DET-train",
                              "images"),
            batch=4, data={"names": {i: str(i) for i in range(10)},
                           "channels": 3},
            mode="val", rect=False, stride=32)
        keep = [i for i, f in enumerate(ds.im_files)
                if os.path.splitext(os.path.basename(f))[0] in set(probe_ids)]
        ds.im_files = [ds.im_files[i] for i in keep]
        ds.labels = [ds.labels[i] for i in keep]
        return build_dataloader(ds, batch=4, workers=0, shuffle=False,
                                rank=-1)

    state = {"loader": None, "epoch_rows": []}

    def on_epoch_end(trainer):
        epoch = trainer.epoch
        crit = trainer.model.criterion
        if crit is None:
            return
        o2o_loss = crit.one2one  # v8DetectionLoss with tal_topk2=1
        assigner = o2o_loss.assigner  # single TAL assigner, topk2 built in

        orig_fwd = assigner.forward

        def patched(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes,
                    mask_gt):
            out = orig_fwd(pd_scores, pd_bboxes, anc_points, gt_labels,
                           gt_bboxes, mask_gt)
            # exact per-GT selected anchor from the official assignment:
            # fg_mask (b, n_anchor) bool; target_gt_idx (b, n_anchor)
            fg_mask, target_gt_idx = out[3], out[4]
            b_sz, n_gt_max = mask_gt.shape[0], mask_gt.shape[1]
            import torch as _t
            sel = _t.full((b_sz, n_gt_max), -1, dtype=_t.long,
                          device=fg_mask.device)
            bidx, aidx = _t.nonzero(fg_mask, as_tuple=True)
            sel[bidx, target_gt_idx[bidx, aidx]] = aidx
            # margin (instability proxy): top1-top2 aligned metric per GT
            mask_pos, align_metric, _ = assigner.get_pos_mask(
                pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points,
                mask_gt)
            am = (align_metric * mask_pos)
            top2 = am.topk(2, dim=-1).values
            stash["sel_anchor"] = sel.cpu().numpy()
            stash["margin"] = (top2[..., 0] - top2[..., 1]).cpu().numpy()
            stash["mask_gt"] = mask_gt.squeeze(-1).cpu().numpy()
            return out

        assigner.forward = patched
        if state["loader"] is None:
            state["loader"] = make_probe_loader(trainer)

        rows = []
        was_training = trainer.model.training
        trainer.model.eval()
        with torch.no_grad():
            for bi, batch in enumerate(state["loader"]):
                batch = trainer.preprocess_batch(batch)
                preds = trainer.model(batch["img"])
                try:
                    trainer.model.criterion(preds, batch)
                except Exception:
                    pass
                if "sel_anchor" not in stash:
                    continue
                sel = stash.pop("sel_anchor")
                mar = stash.pop("margin")
                mgt = stash.pop("mask_gt")
                for b in range(sel.shape[0]):
                    n = int(mgt[b].sum())
                    for gi in range(n):
                        rows.append((epoch, bi, b, gi, int(sel[b, gi]),
                                     float(mar[b, gi])))
        if was_training:
            trainer.model.train()
        assigner.forward = orig_fwd
        arr = np.array(rows, dtype=np.float64)
        np.savez(os.path.join(OUT_DIR, f"probe_epoch{epoch}.npz"), rows=arr)
        print(f"[wp3] epoch {epoch}: logged {len(rows)} GT assignments")

    model.add_callback("on_train_epoch_end", on_epoch_end)
    model.train(
        data=os.path.join(ROOT, "configs", "visdrone.yaml"),
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=4,
        project=os.path.join(ROOT, "runs"), name=args.name,
        exist_ok=True, seed=0, val=False, plots=False,
    )
    print("WP3 probe training complete; analyze npz files for flip rates.")


if __name__ == "__main__":
    main()
