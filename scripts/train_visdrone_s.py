"""Finetune yolo26s on VisDrone at 1280. Run detached."""
import os
ROOT = os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main():
    from ultralytics import YOLO

    m = YOLO(os.path.join(ROOT, "weights", "yolo26s.pt"))
    m.train(
        data=os.path.join(ROOT, "configs", "visdrone.yaml"),
        epochs=80, imgsz=1280, batch=8, device=1, workers=4,
        project=os.path.join(ROOT, "runs"),
        name="ft_visdrone_yolo26s_1280",
        exist_ok=True, seed=0, val=True, plots=False,
    )


if __name__ == "__main__":
    main()
