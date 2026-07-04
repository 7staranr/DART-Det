"""WP5 v1: "no GT left behind" assigner — dead-GT fallback after multi-GT
anchor arbitration.

Measured pathology (wp5_pathology_scan, converged ft_n): 10-15% of crowded
GTs receive ZERO o2o positives because select_highest_overlaps arbitrates
shared anchors to the max-predicted-IoU GT, which can strip a small crowded
GT of its entire candidate pool. Fix: after arbitration (and topk2 squeeze),
any valid GT with no positive anchor gets its best still-unclaimed in-window
anchor (by align metric, fallback distance) — guaranteeing >=1 positive.

Risk measured by the experiment itself: fallback anchors are lower quality;
if noisy supervision hurts, dense recall will not improve.

Trains yolo26n on VisDrone 80ep@1280 (same protocol as ft_n baseline).
"""
import os
ROOT = os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def patch_assigner(floor_frac=0.0):
    """no-orphan fallback. floor_frac>0 (v2): boost the fallback anchor's
    align_metric in-place to floor_frac * that GT's best in-window align, so
    TAL's per-GT score normalization (tal.py:138-142) yields a NON-zero target
    classification score for the rescued GT (v1 left it ~0 -> no gradient,
    which is why v1 did not replicate)."""
    import torch
    from ultralytics.utils.tal import TaskAlignedAssigner

    orig = TaskAlignedAssigner.select_highest_overlaps

    def patched(self, mask_pos, overlaps, n_max_boxes, align_metric):
        target_gt_idx, fg_mask, mask_pos = orig(
            self, mask_pos, overlaps, n_max_boxes, align_metric)
        has_pos = mask_pos.sum(-1) > 0                      # (b, n_gt)
        valid = align_metric.amax(-1) > 0                   # (b, n_gt)
        dead = valid & ~has_pos
        if dead.any():
            taken = mask_pos.sum(-2) > 0                    # (b, n_anchor)
            cand = align_metric * (~taken).unsqueeze(1)
            cand = torch.where(dead.unsqueeze(-1), cand,
                               torch.zeros_like(cand))
            best = cand.argmax(-1)                          # (b, n_gt)
            bidx, gidx = torch.nonzero(dead, as_tuple=True)
            aidx = best[bidx, gidx]
            ok = cand[bidx, gidx, aidx] > 0
            bidx, gidx, aidx = bidx[ok], gidx[ok], aidx[ok]
            mask_pos[bidx, gidx, aidx] = 1.0
            if floor_frac > 0 and len(bidx) > 0:
                # per-GT best achievable align (over ALL in-window anchors)
                gt_max = align_metric.amax(-1)             # (b, n_gt)
                floor = floor_frac * gt_max[bidx, gidx]
                cur = align_metric[bidx, gidx, aidx]
                align_metric[bidx, gidx, aidx] = torch.maximum(cur, floor)
            fg_mask = mask_pos.sum(-2)
            target_gt_idx = mask_pos.argmax(-2)
        return target_gt_idx, fg_mask, mask_pos

    TaskAlignedAssigner.select_highest_overlaps = patched
    tag = f"v2 floor={floor_frac}" if floor_frac > 0 else "v1"
    print(f"[wp5] no-orphan assigner patch active ({tag})")


def main():
    patch_assigner()
    from ultralytics import YOLO

    m = YOLO(os.path.join(ROOT, "weights", "yolo26n.pt"))
    m.train(
        data=os.path.join(ROOT, "configs", "visdrone.yaml"),
        epochs=80, imgsz=1280, batch=8, device=1, workers=4,
        project=os.path.join(ROOT, "runs"),
        name="wp5_norphan_yolo26n_1280",
        exist_ok=True, seed=0, val=True, plots=False,
    )


if __name__ == "__main__":
    main()
