"""WP5 v2 trainer: no-orphan + align-floor. argv: <seed> <device>.
Name encodes seed so multi-seed McNemar can pair against ft_n baselines.
"""
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    seed = int(sys.argv[1])
    device = sys.argv[2]
    floor = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import wp5_train_norphan
    wp5_train_norphan.patch_assigner(floor_frac=floor)
    from ultralytics import YOLO

    suffix = "" if seed == 0 else f"_s{seed}"
    m = YOLO(os.path.join(ROOT, "weights", "yolo26n.pt"))
    m.train(
        data=os.path.join(ROOT, "configs", "visdrone.yaml"),
        epochs=80, imgsz=1280, batch=8, device=device, workers=4,
        project=os.path.join(ROOT, "runs"),
        name=f"wp5v2_yolo26n_1280{suffix}",
        exist_ok=True, seed=seed, val=True, plots=False,
    )


if __name__ == "__main__":
    main()
