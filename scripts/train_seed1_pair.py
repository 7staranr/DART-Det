"""Seed-1 replication pair: ft_n baseline then WP5 no-orphan, sequential on
Protocol identical to seed-0 runs (80ep@1280 batch8)."""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train(name, seed, patch):
    if patch:
        import wp5_train_norphan
        wp5_train_norphan.patch_assigner()
    from ultralytics import YOLO

    m = YOLO(os.path.join(ROOT, "weights", "yolo26n.pt"))
    m.train(
        data=os.path.join(ROOT, "configs", "visdrone.yaml"),
        epochs=80, imgsz=1280, batch=8, device=1, workers=4,
        project=os.path.join(ROOT, "runs"), name=name,
        exist_ok=True, seed=seed, val=True, plots=False,
    )


def main():
    import sys
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    # baseline seed 1 (unpatched) must run in a SEPARATE process from the
    # patched run; this script runs ONE job per invocation.
    job = sys.argv[1]  # "base1" | "wp5s1"
    if job == "base1":
        train("ft_visdrone_yolo26n_1280_s1", 1, patch=False)
    elif job == "wp5s1":
        train("wp5_norphan_yolo26n_1280_s1", 1, patch=True)


if __name__ == "__main__":
    main()
