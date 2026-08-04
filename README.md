# DART-Det — Density-Adaptive Rank-Truncation for NMS-free detection

Diagnosis-and-repair code for the fixed decode budget of end-to-end, NMS-free detectors. An NMS-free head keeps only the top-K ranked predictions per image; as scene density grows, true positives the network has already scored are pushed past rank K and discarded before any confidence thresholding. We treat this fixed top-K stage as a rate-constrained ranked-selection channel whose effective capacity -- the fraction of objects the budget covers, K_eff/|G| = R@K, not the absolute count K_eff -- collapses as density grows, name the loss Rank-Truncation Information Loss (RTIL = R@M − R@K), decompose the density-stratified recall drop into a recoverable budget-truncation term, a shared-difficulty term that the one-to-many reference path also fails to recover at the cached rank depth, and a small residual between the two evaluated decode paths, and repair the recoverable term at inference by raising the decode cap on dense images, a training-free change; a density gate (DABA, Density-Adaptive Budget Allocation) reaches a numerically similar gain while returning a smaller set on sparse images (see Notes for the exact comparison, which is not an accuracy advantage). That third term is a net path difference rather than a head-only quantity: the two paths differ in head, supervision, assignment, calibration, NMS and matching convention, so its small size does not bound the one-to-one head's isolated contribution. An inference-time recoverability condition predicts recoverability from detector structure: soft top-K heads that cache a rank tail (M > K, e.g. YOLO26, YOLOv10) are recoverable; hard fixed-query budgets (RT-DETR at its deployed 300-query configuration, M ≡ K) are not.

This repository contains the diagnostic protocol, the density-stratified evaluation, DABA, and the figure/table generators. It does not ship datasets; those are public and configured through `DART_ROOT` (see Notes).

## Requirements

```
pip install -r requirements.txt
```

Python 3.11+. The detectors run on a recent Ultralytics release with YOLO26 support (`ultralytics`), PyTorch with CUDA, plus `numpy`, `scipy`, `pandas`, `statsmodels` (image-clustered regression), `matplotlib`, `opencv-python`, and `Pillow`. A single 10–20 GB GPU is sufficient for every finetune. Cache-based evaluation and statistical reduction are CPU-only; finetuning, prediction-cache generation, the masking intervention, assignment instrumentation, and the latency benchmark all execute the detector and normally need a GPU.

Set the workspace root once (defaults to the repository directory):

```
export DART_ROOT=/path/to/workspace     # bash
$env:DART_ROOT = "C:\path\to\workspace" # PowerShell
```

Scripts read `DART_ROOT/data`, write finetunes to `DART_ROOT/runs`, and write cached predictions and analysis to `DART_ROOT/experiments`.

## Repository layout

```
scripts/
  # data preparation
  visdrone_to_yolo.py        VisDrone-DET annotations -> YOLO labels
  sku110k_to_yolo.py         SKU-110K CSV -> YOLO labels + split lists
  prep_dota.py               DOTAv1 -> tiled horizontal-box detection set
  make_dense_val.py          dense-image (GT >= k) val lists + data yamls
  gt_density_stats.py        per-image object-count statistics per split

  # finetuning (soft top-K and hard-query detectors)
  train_visdrone_n.py / _s.py / _n_gpu1.py     YOLO26-n/s on VisDrone @1280
  train_sku_n.py / _n_seed.py                  YOLO26-n on SKU-110K @1024
  train_dota.py                                YOLO26-n on DOTAv1-tiled
  train_yolov10_visdrone.py                    YOLOv10-n (2nd soft head)
  train_rtdetr_visdrone.py / _sku.py           RT-DETR-L (hard query budget)
  train_seed1_pair.py, resume_sku_gpu1.py      seed replication / resume
  run_wp1.ps1, run_wp1_ft.ps1                  end-to-end pipeline drivers

  # stage 1 - diagnosis: density-stratified recall-at-budget
  wp1_infer.py               cache rank-resolved predictions (depth 1000)
  wp1_eval.py                density-stratified recall + budget statistics
  wp1_plot.py                recall-density and budget-saturation curves
  wp1_fppi.py                FPPI-conditioned recall (deployment-realistic)
  wp1_local_density.py       local-crowding vs scene-cardinality discriminator
  wp1_slot_occupant.py       top-300 slot composition (distinct-TP/dup/FP)
  diag_glmm.py               image-clustered logistic regression of detection
  d1_verify_yolo26.py, d1b_verify_paths.py     inference-semantics checks

  # stage 2 - causal context-masking intervention
  wp2_mask_intervention.py   gray-fill outside a 3x probe window, score lift vs placebo
  wp2_mask_bootstrap.py      image-clustered bootstrap, miss-subgroup

  # stage 3 - training-time assignment dynamics
  wp3_assign_dynamics.py     assignment-flip instrumentation during training
  wp3_analyze_flips.py       flip rate and stability margin vs crowding
  wp3_harm_link.py           do assignment-unstable GTs end up worse detected?

  # stage 4 - budget repair (DABA) and its evaluation
  wp4_budget_policy.py       DABA: density-to-budget allocation (the repair)
  wp4_gate_probe.py          pre-specified decision-gate probe (no training)
  wp4_ap_eval.py             does relaxing the budget raise or lower AP?
  wp4_ap_per_bucket.py       per-density-bucket AP vs budget
  wp4_ap_bootstrap.py        image-bootstrap CI on the dense-subset AP gain
  daba_gate_ap.py            AP realized by the DABA gate itself (per-image K*)
  decomp_endpoint_check.py   decomposition robustness to the dense endpoint
  wp4_conf_floor.py          deploy-threshold sensitivity of the recovery
  wp4_latency_bench.py       decode-cost / latency benchmark
  scale_stratified_ap.py     per-object-scale AP (very-tiny ... large)
  recoverability_table.py    soft-vs-hard recoverability table (M, R@300, R@1000)
  lambda_bracket.py          cost-model lambda bracket from cached predictions
  nms_insensitivity.py       one-to-many+NMS AP across NMS IoU thresholds

  # stage 5 - training-time pathology and the negative result
  wp5_pathology_scan.py      candidate-pool collapse in crowded regions
  wp5_train_norphan.py       "no GT left behind" assigner (v1)
  wp5_train_v2.py            no-orphan + align-floor assigner (v2)
  wp5_mcnemar.py             paired McNemar test vs baseline
  wp5_mcnemar_clustered.py   image-block permutation McNemar + BH-FDR

  # figures and tables
  paper_figures.py           main result figures
  fig_capacity_curve.py      recall-at-budget R@k vs k (soft rises, hard flat)
  fig_recoverable_mass.py    recoverable-mass profile b(n)
  fig_dart.py                framework schematic
  fig_qualitative.py         qualitative cap-300 vs cap-1000 on a dense image
  iteration1_tables.py       cache-based result tables

configs/
  visdrone.yaml, visdrone_dense.yaml, sku110k.yaml, sku110k_dense.yaml,
  dota.yaml, _apbucket_*.yaml, _sparse_*.yaml     Ultralytics data configs

data/splits/                                      the density splits themselves
  SKU110K_fixed/       bucket_{50-100,100-150,150-300,ge300}.txt (101/1792/
                       1004/30 images) and dense_test.txt (1034)
  VisDrone2019-DET-val/bucket_{50-100,100-150,150-300}.txt (248/81/27)
                       and dense.txt (30)

results/buckets/                                  per-bucket recall tables
  wp1_pilot/, wp1_ft/, wp1_sku/, wp1_rtdetr/      (54 CSVs)
results/per_image/                                per-image matched@K, cache
  wp1_pilot/, wp1_ft/, wp1_sku/, wp1_rtdetr/      depth and ndet@conf (54 CSVs)
results/per_gt/wp1_ft/                            per-ground-truth outcomes:
                                                  area, max_iou_nbr, nbr_count,
                                                  rank, m300, m1000 (7 CSVs) --
                                                  the input to diag_glmm.py and
                                                  wp5_mcnemar_clustered.py
results/masking/wp2_mask/                         context-masking probe records
                                                  (dense + placebo arms)

weights_release/                                  the three finetunes carrying
  visdrone_yolo26n_1280.pt   (5.2 MB, mAP@0.5 0.449)   the primary claims, so the
  visdrone_yolo26s_1280.pt   (19.4 MB, mAP@0.5 0.530)  cache-dependent analyses
  sku110k_yolo26n_1024.pt    (5.2 MB, mAP@0.5 0.902)   can be rerun without
                                                       refinetuning

training_manifest.csv                             what `optimizer=auto` actually
                                                  resolved to per run, read off
                                                  the training logs
```

### What `optimizer=auto` resolved to

`args.yaml` records `optimizer: auto` and `lr0: 0.01` — the *inputs*, not what
Ultralytics chose. `training_manifest.csv` records the resolved optimizer and
effective initial learning rate as printed by the trainer:

| run | optimizer | lr0 |
|---|---|---|
| ft_visdrone_yolo26n_1280 | AdamW | 7.14e-4 |
| ft_visdrone_yolo26s_1280 | AdamW | 7.14e-4 |
| ft_visdrone_yolov10n_1280 | AdamW | 7.14e-4 |
| ft_visdrone_rtdetr_l_960 | AdamW | 7.14e-4 |
| ft_sku110k_yolo26n_1024 (+ `_s1`) | AdamW | 2e-3 |
| ft_sku_rtdetr_l_1024 | AdamW | 2e-3 |
| ft_dota_yolo26n_1024 | MuSGD | 0.01 |

`cos_lr: false` throughout, so the schedule is the framework's linear decay, not
cosine.

### Regenerating the prediction caches

The rank-resolved caches are ~1.1 GB and are not tracked, but the checkpoints
that produce them are. With the public imagery in place under `DART_ROOT/data`:

```bash
python scripts/wp1_infer.py --model weights_release/visdrone_yolo26s_1280.pt \
    --list data/splits/VisDrone2019-DET-val/dense.txt \
    --out experiments/wp1_ft/preds_visdrone_ft_s.jsonl \
    --imgsz 1280 --max-det 1000
```

That is the step every cache-dependent table and figure depends on; the
analysis scripts read the resulting `.jsonl` directly.

Header note: 11 of the 54 bucket tables use `AR@k` column names (the earlier
COCO-pretrained pass) and 43 use `R@k`. The split is by table, not by
directory: `results/buckets/wp1_pilot/` holds 19 files, 8 of which (`*_deploy`, `crowdhuman_yolo26n_{fbox,vbox}`) already carry the `R@k`
columns. Key on whichever prefix the header actually has.

`data/splits/` and the four `results/` trees are checked in, so the density
stratification and every per-bucket, per-image and per-ground-truth number in
the paper can be inspected without regenerating anything.

What a clean clone can actually run, by level:

| Level | Needs | Examples |
|---|---|---|
| 1. CSV reduction | nothing but this repo | `decomp_endpoint_check.py`, `recoverability_table.py`, `daba_edge_sweep.py --per-image results/per_image/...` |
| 2. Cache-consuming analysis | the public imagery + a released checkpoint, then `wp1_infer.py` to rebuild the ~1.1 GB JSONL caches | `wp1_eval.py`, `wp4_budget_policy.py`, `wp4_ap_bootstrap.py`, `wp4_conf_floor.py`, `daba_gate_ap.py`, `lambda_bracket.py` |
| 3. Re-training | a GPU; only three checkpoints are deposited (VisDrone YOLO26-n/s, SKU-110K YOLO26-n), so DOTA, YOLOv10, RT-DETR and the second seeds must be retrained from the `train_*.py` scripts | `train_dota.py`, `train_yolov10_visdrone.py`, `train_rtdetr_*.py`, `train_seed1_pair.py` |
| 4. GPU interventions | a GPU and the detector in the loop | `wp2_mask_intervention.py`, `wp3_assign_dynamics.py`, `wp4_latency_bench.py` |

Level 1 needs no download:

```bash
python scripts/daba_edge_sweep.py \
    --per-image results/per_image/wp1_sku/sku_test_ft_per_image.csv
python scripts/decomp_endpoint_check.py
python scripts/recoverability_table.py
```

The per-image rank-resolved prediction *caches* (the raw ranked detections) are
~1.1 GB and are **not** checked in; `scripts/wp1_infer.py` regenerates them
(see "Reproducing the paper" below), after which the analysis scripts reproduce
the tables above.

The `wp1`–`wp5` prefixes group the scripts by analysis stage: `wp1` = diagnosis, `wp2` = causal masking, `wp3` = assignment dynamics, `wp4` = budget repair, `wp5` = training-time pathology.

## DABA in one place

The repair is inference-only, and it is an *output-selection* policy rather than adaptive computation. Every evaluated image is first decoded to depth M=1000. DABA counts detections above a low proxy floor **on that already-decoded list**, then returns a prefix K*(x). It does not reduce backbone, head, or depth-1000 decode time; what it resizes is the returned detection set:

```
n = number of detections with score >= 0.1        # density proxy
K = 300   if n < 100
    600   if n < 200
    1000  otherwise
return the cached ranked list re-truncated to top-K
```

No second inference, no retraining, no added module. The implementation is `scripts/wp4_budget_policy.py`; it acts on any soft top-K head that caches a rank tail (M > K).

## Quick start

DABA is inference-only. Given a rank-cached prediction file (produced by
`wp1_infer.py`; see Reproducing), apply the density-to-budget rule and score the
recovered recall:

```
python scripts/wp4_budget_policy.py --dataset sku --gt <SKU110K_fixed/annotations/annotations_test.csv> --preds <preds.jsonl>
```

Seven scripts share one interface -- `wp1_eval.py`, `wp1_fppi.py`,
`wp1_slot_occupant.py`, `wp4_budget_policy.py`, `wp4_ap_bootstrap.py`,
`wp4_conf_floor.py` and `daba_gate_ap.py` -- taking `--dataset`, `--gt`
(ground-truth labels) and `--preds` (the cached prediction JSONL).
`lambda_bracket.py` takes per-split paths instead and `recoverability_table.py`
takes no arguments at all. Run any argument-taking script with `--help` for
its full options.

## Reproducing the paper

Set `DART_ROOT` first and place the datasets under `DART_ROOT/data` (see Data
sources). Paths below are illustrative placeholders. The PowerShell drivers wire
the full per-model sequence (infer -> density-stratified eval -> plots); the
steps below show the underlying interface, and the argument-taking scripts accept `--help`.

```
# 0. datasets -> YOLO format + density-stratified val lists
python scripts/visdrone_to_yolo.py --root data/VisDrone2019-DET-train
python scripts/sku110k_to_yolo.py
python scripts/prep_dota.py
python scripts/make_dense_val.py
python scripts/gt_density_stats.py

# 1. finetune the detectors. Only the VisDrone YOLO26-n and SKU-110K YOLO26-n
#    finetunes (and the two wp5 assigner variants) have a second seed; the
#    *_seed / *_v2 variants take the seed as argv, others fix seed=0.
python scripts/train_visdrone_n.py                   # YOLO26-n on VisDrone
python scripts/train_sku_n_seed.py 0                 # YOLO26-n on SKU-110K, seed 0
python scripts/train_dota.py                         # 2nd aerial domain
python scripts/train_yolov10_visdrone.py             # 2nd soft top-K head
python scripts/train_rtdetr_visdrone.py              # hard query budget

# 2. cache rank-resolved predictions (depth 1000), then the diagnosis.
#    run_wp1_ft.ps1 wires wp1_infer -> wp1_eval -> wp1_plot for one model:
./scripts/run_wp1_ft.ps1 -Weights runs/<run>/weights/best.pt -Tag ft_n -Device 0
#    or call the stages directly:
python scripts/wp1_infer.py --model runs/<run>/weights/best.pt \
    --list data/VisDrone2019-DET-val/val.txt --out experiments/preds_ft_n.jsonl
python scripts/wp1_eval.py --dataset visdrone --gt data/VisDrone2019-DET-val/annotations \
    --preds experiments/preds_ft_n.jsonl --out-prefix experiments/ft_n
python scripts/wp1_slot_occupant.py --dataset visdrone \
    --gt data/VisDrone2019-DET-val/annotations --preds experiments/preds_ft_n.jsonl
python scripts/wp1_local_density.py --dataset visdrone \
    --gt data/VisDrone2019-DET-val/annotations --preds experiments/preds_ft_n.jsonl \
    --out-prefix experiments/ft_n          # emits ft_n_per_gt.csv
python scripts/diag_glmm.py --per-gt experiments/ft_n_per_gt.csv

# 3. causal context-masking intervention
python scripts/wp2_mask_intervention.py --model runs/<run>/weights/best.pt \
    --gt data/VisDrone2019-DET-val/annotations --images data/VisDrone2019-DET-val/images \
    --out experiments/mask.json

# 4. budget repair (DABA) + AP, precision, recoverability
# --gt is dataset-specific: VisDrone = the comma-separated annotations dir,
# SKU-110K = the annotation CSV, DOTA = a YOLO labels dir.
SKU_GT=data/SKU110K_fixed/annotations/annotations_test.csv
python scripts/wp4_budget_policy.py --dataset sku --gt $SKU_GT --preds <preds.jsonl>
python scripts/wp4_ap_bootstrap.py --dataset sku --gt $SKU_GT --preds <preds.jsonl> --thr-density 150
python scripts/wp4_conf_floor.py   --dataset sku --gt $SKU_GT --preds <preds.jsonl>
python scripts/recoverability_table.py

# 5. training-time pathology + negative result (two seeds)
python scripts/wp5_pathology_scan.py
python scripts/wp5_train_v2.py 0 0                    # seed 0, device 0
#    --pairs takes (baseline, treatment) per-GT tables, two per seed:
python scripts/wp5_mcnemar_clustered.py --pairs \
    results/per_gt/wp1_ft/visdrone_ft_n_local_per_gt.csv \
    results/per_gt/wp1_ft/visdrone_wp5v2_local_per_gt.csv \
    results/per_gt/wp1_ft/visdrone_ft_n_s1_local_per_gt.csv \
    results/per_gt/wp1_ft/visdrone_wp5v2_s1_local_per_gt.csv
```

## Figures and statistics

```
python scripts/paper_figures.py          # recall-density, slot composition, AP, mask
python scripts/fig_capacity_curve.py     # R@k vs k: soft rises, hard flat
python scripts/fig_recoverable_mass.py   # boundary-region size b(n)
python scripts/fig_qualitative.py        # cap-300 vs cap-1000 on a dense shelf
python scripts/scale_stratified_ap.py    # per-object-scale AP table
python scripts/lambda_bracket.py         # cost-model bracket
```

## Data sources

All datasets are public and used under their own licenses; download them into `DART_ROOT/data`.

- VisDrone-DET — Zhu et al., aerial detection benchmark.
- SKU-110K — Goldman et al., densely packed retail shelves.
- DOTA v1.0 — Xia et al., aerial images; tiled to horizontal-box crops here as a second aerial domain (not the oriented-box leaderboard).
- CrowdHuman — Shao et al., person-class low-domain-gap pilot.
- MS-COCO — Lin et al., domain-gap pilots only.

## Notes

- Recall-at-budget `R@K` is the fraction of ground-truth objects matched (class-agnostic, IoU >= 0.5 unless stated) within the top-K ranked predictions the deployed head emits. It is read at any budget by offline truncation of a single depth-1000 cache. Because top-K selection is order-preserving and never enters the training loss, truncating the cache reproduces the ranked prefix that re-running at that budget would emit. The cache stores boxes rounded to 0.1 px and confidences to 4 decimals, so downstream metrics are reproduced to that quantization rather than bit-for-bit.
- The nominal budget is `K_nom = 300`; the effective budget `K_eff` is the number of distinct true positives the kept slots carry, which falls below `K_nom` when duplicates and false positives occupy slots.
- Evaluation is deploy-faithful: ignore regions are exempted after truncation, not before; matching is greedy in descending confidence at the deployed operating thresholds.
- The VisDrone and SKU-110K *diagnosis* numbers (YOLO26-n) and the SKU-110K *repair* gain are reported over two seeds. The VisDrone repair (YOLO26-s), DOTA, and YOLOv10 results are single-run: only `ft_visdrone_yolo26n_1280`, `ft_sku110k_yolo26n_1024`, `wp5_norphan_yolo26n_1280` and `wp5v2_yolo26n_1280` have `_s1` twins.
- Every path is relative to `DART_ROOT` (env var; defaults to the repo root). No absolute paths are hard-coded.

## Citation

Prepared for submission to MDPI *AI*. `CITATION.cff` carries the machine-readable
metadata (GitHub renders it as "Cite this repository"); the BibTeX below will
gain volume/pages/DOI on acceptance:

```bibtex
@article{dart_rtil,
  title   = {DART: Diagnosing and Repairing Rank-Truncation Information Loss in Fixed-Budget End-to-End Detection},
  author  = {Yang, Jixiang and Yi, Junfei and Li, Jinhan and Ma, Shengjie},
  year    = {2026},
  note    = {Manuscript under review}
}
```

## Authors

| | Affiliation | ORCID |
|---|---|---|
| Jixiang Yang | Qingdao University of Science and Technology | [0009-0002-7291-8366](https://orcid.org/0009-0002-7291-8366) |
| Junfei Yi | The University of Manchester | [0009-0008-0063-4085](https://orcid.org/0009-0008-0063-4085) |
| Jinhan Li | Qingdao University of Science and Technology | — |
| Shengjie Ma (corresponding) | Qingdao University of Science and Technology | [0000-0001-7506-8120](https://orcid.org/0000-0001-7506-8120) |

Correspondence: mashengjie@qust.edu.cn
