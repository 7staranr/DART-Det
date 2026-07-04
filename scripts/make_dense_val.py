"""Emit dense-image val lists (GT>=threshold) + yamls for stratified AP.

The fixed-300 budget binds only on dense images; global AP is dominated by
medium-density images and is ~flat in max_det. To measure the budget repair's
AP value we evaluate on a dense-only subset where the cap actually binds.
"""
import os

import wp1_eval as we

ROOT = os.environ.get("DART_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def emit(name, img_dir, ids, yaml_names, yaml_path, list_path):
    with open(list_path, "w", encoding="utf-8") as f:
        f.write("\n".join(os.path.join(img_dir, i) for i in ids) + "\n")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {ROOT}\nval: {list_path}\ntrain: {list_path}\n\n")
        f.write("names:\n")
        for k, v in yaml_names.items():
            f.write(f"  {k}: {v}\n")
    print(f"{name}: {len(ids)} dense images -> {list_path}")


def main():
    thr = 150
    # VisDrone val
    vd = we.load_visdrone_gt(
        os.path.join(ROOT, "data", "VisDrone2019-DET-val", "annotations"))
    vd_ids = [f"{k}.jpg" for k, v in vd.items() if len(v["gt"]) >= thr]
    emit("VisDrone-val", os.path.join(ROOT, "data",
         "VisDrone2019-DET-val", "images"), vd_ids,
         {i: n for i, n in enumerate(
             ["pedestrian", "people", "bicycle", "car", "van", "truck",
              "tricycle", "awning-tricycle", "bus", "motor"])},
         os.path.join(ROOT, "configs", "visdrone_dense.yaml"),
         os.path.join(ROOT, "data", "VisDrone2019-DET-val", "dense.txt"))

    # SKU test
    sku = we.load_sku_gt(
        os.path.join(ROOT, "data", "SKU110K_fixed", "annotations",
                     "annotations_test.csv"))
    sku_ids = [f"{k}.jpg" for k, v in sku.items() if len(v["gt"]) >= thr]
    emit("SKU-test", os.path.join(ROOT, "data", "SKU110K_fixed", "images"),
         sku_ids, {0: "object"},
         os.path.join(ROOT, "configs", "sku110k_dense.yaml"),
         os.path.join(ROOT, "data", "SKU110K_fixed", "dense_test.txt"))


if __name__ == "__main__":
    main()
