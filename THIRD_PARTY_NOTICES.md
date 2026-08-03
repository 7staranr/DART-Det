# Third-party notices

The MIT `LICENSE` in this repository covers **the code written for this project**.
It does not, and cannot, relicense the datasets, the upstream detector
implementations, or the pretrained weights those checkpoints were finetuned
from. This file states what each released artifact derives from, so that anyone
redistributing it can check the upstream terms directly rather than inferring
them from the repository licence.

## Datasets

No dataset imagery or annotation file is redistributed here. `data/splits/`
contains only **lists of image identifiers**, which select subsets of the public
releases below; you must obtain the data from its original source under its own
terms.

| Dataset | Used for | Obtain from |
|---|---|---|
| VisDrone2019-DET | primary aerial domain (val, test-dev) | the official VisDrone challenge release |
| SKU-110K (`SKU110K_fixed`) | dense-tail stress test | the authors' official release |
| DOTA v1.0 | second aerial domain, tiled locally to horizontal boxes | the official DOTA release |
| CrowdHuman | low-domain-gap pilot only | the official CrowdHuman release |
| MS-COCO | pretraining provenance of the base weights only | the official COCO release |

Each is distributed under its own licence and, in several cases, is restricted
to non-commercial or academic use. Check the terms that apply to your use before
redistributing anything derived from them.

**DOTA is not used as the official benchmark.** It is tiled here to 1024-px
crops and evaluated as *horizontal-box* detection scored per tile. Those numbers
are not comparable to the official oriented-box leaderboard and are not
presented as such.

## Detector implementations

The finetunes and all inference run through the Ultralytics framework
(YOLO26, YOLOv10 and RT-DETR entry points). Ultralytics is distributed under
AGPL-3.0, which carries obligations of its own for downstream use and for
network-served deployments. This repository does not vendor any Ultralytics
source; it imports the pinned release listed in `requirements.txt`.

## Released checkpoints (`weights_release/`)

| File | Base weights | Finetuned on | Reported in |
|---|---|---|---|
| `visdrone_yolo26n_1280.pt` | YOLO26-n (COCO-pretrained, Ultralytics) | VisDrone2019-DET-train @1280 | VisDrone diagnosis, mAP@0.5 0.449 |
| `visdrone_yolo26s_1280.pt` | YOLO26-s (COCO-pretrained, Ultralytics) | VisDrone2019-DET-train @1280 | VisDrone diagnosis and repair, mAP@0.5 0.530 |
| `sku110k_yolo26n_1024.pt` | YOLO26-n (COCO-pretrained, Ultralytics) | SKU-110K train @1024 | SKU-110K repair, mAP@0.5 0.902 |

These are derivative works of the upstream pretrained weights and of the
datasets they were finetuned on; whatever terms attach upstream continue to
apply. They are released to make the reported numbers checkable, not as
generally licensed models.

The checkpoints deposited here have had their `train_args` path metadata
normalized to repository-relative values. Only that metadata was rewritten: no
tensor, EMA buffer, class name, or other inference-relevant field was touched,
and the sanitized checkpoints were verified to load and produce byte-identical
predictions. See `weights_release/SHA256SUMS` for the current checksums.
