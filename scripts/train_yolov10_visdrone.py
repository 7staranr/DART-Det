"""YOLOv10-n finetune on VisDrone (second soft top-K detector).
YOLOv10 also uses a one-to-one head with a soft top-K decode budget, so it
tests whether the max_det repair transfers across the soft-budget CLASS, not
just YOLO26."""
import os
ROOT = os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main():
    from ultralytics import YOLO
    m = YOLO("yolov10n.pt")
    m.train(
        data=os.path.join(ROOT, "configs", "visdrone.yaml"),
        epochs=80, imgsz=1280, batch=6, device=0, workers=4,
        project=os.path.join(ROOT, "runs"),
        name="ft_visdrone_yolov10n_1280", exist_ok=True, seed=0,
        val=True, plots=False)


if __name__ == "__main__":
    main()
