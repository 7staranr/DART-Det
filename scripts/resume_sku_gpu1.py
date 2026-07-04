"""Resume SKU-110K finetune after an out-of-memory restart."""
import os
ROOT = os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main():
    from ultralytics import YOLO

    last = os.path.join(ROOT, "runs", "ft_sku110k_yolo26n_1024",
                        "weights", "last.pt")
    m = YOLO(last)
    m.train(resume=True, device=1)


if __name__ == "__main__":
    main()
