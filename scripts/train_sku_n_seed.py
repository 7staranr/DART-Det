"""SKU-110K yolo26n finetune, seed via argv (seed-robustness check).
argv: <seed> <device>."""
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    seed, device = int(sys.argv[1]), sys.argv[2]
    from ultralytics import YOLO
    suffix = "" if seed == 0 else f"_s{seed}"
    m = YOLO(os.path.join(ROOT, "weights", "yolo26n.pt"))
    m.train(data=os.path.join(ROOT, "configs", "sku110k.yaml"),
            epochs=60, imgsz=1024, batch=6, device=device, workers=4,
            project=os.path.join(ROOT, "runs"),
            name=f"ft_sku110k_yolo26n_1024{suffix}",
            exist_ok=True, seed=seed, val=True, plots=False)


if __name__ == "__main__":
    main()
