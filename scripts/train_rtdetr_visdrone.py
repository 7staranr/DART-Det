"""RT-DETR-L finetune on VisDrone.

RT-DETR's prediction budget is architecturally HARD (300 decoder queries),
unlike YOLO26's soft topk over dense anchors — the diagnosis contrast itself
is a paradigm-level observation.
"""
import os
ROOT = os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main():
    from ultralytics import RTDETR

    m = RTDETR("rtdetr-l.pt")  # auto-downloads
    m.train(
        data=os.path.join(ROOT, "configs", "visdrone.yaml"),
        epochs=60, imgsz=960, batch=6, device=1, workers=4,
        mosaic=0.0,  # RT-DETR recipe has no mosaic; ultralytics RTDETRDataset
        # mosaic canvas (2x imgsz=1920) exceeded GPU memory
        project=os.path.join(ROOT, "runs"),
        name="ft_visdrone_rtdetr_l_960",
        exist_ok=True, seed=0, val=True, plots=False,
    )


if __name__ == "__main__":
    main()
