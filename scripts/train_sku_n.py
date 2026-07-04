"""SKU-110K finetune: yolo26n, single class.

imgsz 1024 (products are medium-size),
60 epochs. Purpose: in-domain weights on the only benchmark with a real >300
density population (val max 718) -> hard-H1 saturation test + density curves.
"""
import os
ROOT = os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main():
    from ultralytics import YOLO

    m = YOLO(os.path.join(ROOT, "weights", "yolo26n.pt"))
    m.train(
        data=os.path.join(ROOT, "configs", "sku110k.yaml"),
        epochs=60, imgsz=1024, batch=6, device=0, workers=4,
        project=os.path.join(ROOT, "runs"),
        name="ft_sku110k_yolo26n_1024",
        exist_ok=True, seed=0, val=True, plots=False,
    )


if __name__ == "__main__":
    main()
