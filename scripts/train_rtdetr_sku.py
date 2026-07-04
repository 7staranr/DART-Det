"""RT-DETR-L finetune on SKU-110K, imgsz 1024 to match
the YOLO26-SKU run (ft_sku110k_yolo26n_1024) for a fair cross-family
dense-subset comparison. mosaic=0 (RTDETRDataset mosaic canvas = 2x imgsz
OOMs; also closer to DETR recipe). Single class."""
import os
ROOT = os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main():
    from ultralytics import RTDETR

    m = RTDETR("rtdetr-l.pt")
    m.train(
        data=os.path.join(ROOT, "configs", "sku110k.yaml"),
        epochs=60, imgsz=1024, batch=6, device=1, workers=4, mosaic=0.0,
        project=os.path.join(ROOT, "runs"),
        name="ft_sku_rtdetr_l_1024",
        exist_ok=True, seed=0, val=True, plots=False,
    )


if __name__ == "__main__":
    main()
