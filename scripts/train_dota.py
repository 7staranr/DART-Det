"""YOLO26-n finetune on DOTAv1-tiled (second aerial domain).
Provides a second aerial dense benchmark (distinct sensor/scenes from VisDrone)
to test the recall-density decline and budget repair across aerial domains.
"""
import os
ROOT = os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main():
    from ultralytics import YOLO
    m = YOLO("yolo26n.pt")
    m.train(
        data=os.path.join(ROOT, "configs", "dota.yaml"),
        epochs=60, imgsz=1024, batch=8, device=1, workers=4,
        project=os.path.join(ROOT, "runs"),
        name="ft_dota_yolo26n_1024", exist_ok=True, seed=0,
        val=True, plots=False, close_mosaic=10)


if __name__ == "__main__":
    main()
