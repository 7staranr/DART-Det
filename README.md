# DART-Det — Density-Adaptive Rank-Truncation for NMS-free detection

Diagnosis-and-repair code for the fixed decode budget of end-to-end, NMS-free detectors. An NMS-free head keeps only the top-K ranked predictions per image; as scene density grows, true positives the network has already scored are pushed past rank K and discarded before any confidence thresholding. We treat this fixed top-K stage as a rate-constrained ranked-selection channel whose effective capacity K_eff collapses below the nominal budget K_nom = 300, name the loss Rank-Truncation Information Loss (RTIL = R@M − R@K), decompose the density-stratified recall drop into a recoverable budget-truncation term, an irreducible shared-difficulty term, and a small one-to-one-head residual, and repair the recoverable term at inference with DABA (Density-Adaptive Budget Allocation), a training-free density-to-budget rule. A soft-versus-hard transfer law predicts recoverability from detector structure: soft top-K heads that cache a rank tail (M > K, e.g. YOLO26, YOLOv10) are recoverable; hard fixed-query budgets (RT-DETR, M ≡ K) are not.

This repository contains the diagnostic protocol, the density-stratified evaluation, DABA, and the figure/table generators. It does not ship datasets or weights; both are public and configured through `DART_ROOT` (see Notes).

## Requirements

```
pip install -r requirements.txt
```

Python 3.10+. The detectors run on a recent Ultralytics release with YOLO26 support (`ultralytics`), PyTorch with CUDA, plus `numpy`, `scipy`, `pandas`, `statsmodels` (image-clustered regression), `matplotlib`, `opencv-python`, and `Pillow`. A single 10–20 GB GPU is sufficient for every finetune; all analysis is CPU-only and reads cached predictions.

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
  wp2_mask_intervention.py   remove competitors, measure score lift vs placebo
  wp2_mask_bootstrap.py      image-clustered bootstrap, miss-subgroup

  # stage 3 - training-time assignment dynamics
  wp3_assign_dynamics.py     assignment-flip instrumentation during training
  wp3_analyze_flips.py       flip rate and stability margin vs crowding
  wp3_harm_link.py           do assignment-unstable GTs end up worse detected?

  # stage 4 - budget repair (DABA) and its evaluation
  wp4_budget_policy.py       DABA: density-to-budget allocation (the repair)
  wp4_gate_probe.py          pre-registered decision-gate probe (no training)
  wp4_ap_eval.py             does relaxing the budget raise or lower AP?
  wp4_ap_per_bucket.py       per-density-bucket AP vs budget
  wp4_ap_bootstrap.py        image-bootstrap CI on the dense-subset AP gain
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
```

The `wp1`–`wp5` prefixes group the scripts by analysis stage: `wp1` = diagnosis, `wp2` = causal masking, `wp3` = assignment dynamics, `wp4` = budget repair, `wp5` = training-time pathology.

## DABA in one place

The repair is inference-only. From a single cached forward pass, count detections above a low proxy floor and map the count to a decode budget:

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
python scripts/wp4_budget_policy.py --dataset sku --gt <labels_dir> --preds <preds.jsonl>
```

The diagnosis and every analysis script share the same interface: `--dataset`,
`--gt` (ground-truth labels), and `--preds` (the cached prediction JSONL). Run
any script with `--help` for its full options.

## Reproducing the paper

Set `DART_ROOT` first and place the datasets under `DART_ROOT/data` (see Data
sources). Paths below are illustrative placeholders. The PowerShell drivers wire
the full per-model sequence (infer -> density-stratified eval -> plots); the
steps below show the underlying interface, and every script accepts `--help`.

```
# 0. datasets -> YOLO format + density-stratified val lists
python scripts/visdrone_to_yolo.py --root data/VisDrone2019-DET-train
python scripts/sku110k_to_yolo.py
python scripts/prep_dota.py
python scripts/make_dense_val.py
python scripts/gt_density_stats.py

# 1. finetune the detectors. Primary VisDrone/SKU results use two seeds:
#    the *_seed / *_v2 variants take the seed as argv; others fix seed=0.
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
python scripts/wp1_eval.py --dataset visdrone --gt data/VisDrone2019-DET-val/labels \
    --preds experiments/preds_ft_n.jsonl --out-prefix experiments/ft_n
python scripts/wp1_slot_occupant.py --dataset visdrone \
    --gt data/VisDrone2019-DET-val/labels --preds experiments/preds_ft_n.jsonl
python scripts/wp1_local_density.py --dataset visdrone \
    --gt data/VisDrone2019-DET-val/labels --preds experiments/preds_ft_n.jsonl \
    --out-prefix experiments/ft_n          # emits ft_n_per_gt.csv
python scripts/diag_glmm.py --per-gt experiments/ft_n_per_gt.csv

# 3. causal context-masking intervention
python scripts/wp2_mask_intervention.py --model runs/<run>/weights/best.pt \
    --gt data/VisDrone2019-DET-val/labels --images data/VisDrone2019-DET-val/images \
    --out experiments/mask.json

# 4. budget repair (DABA) + AP, precision, recoverability
python scripts/wp4_budget_policy.py --dataset sku --gt <labels> --preds <preds.jsonl>
python scripts/wp4_ap_bootstrap.py --dataset sku --gt <labels> --preds <preds.jsonl> --thr-density 150
python scripts/wp4_conf_floor.py   --dataset sku --gt <labels> --preds <preds.jsonl>
python scripts/recoverability_table.py

# 5. training-time pathology + negative result (two seeds)
python scripts/wp5_pathology_scan.py
python scripts/wp5_train_v2.py 0 0                    # seed 0, device 0
python scripts/wp5_mcnemar_clustered.py --pairs experiments/mcnemar_pairs.jsonl
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

- Recall-at-budget `R@K` is the fraction of ground-truth objects matched (class-agnostic, IoU >= 0.5 unless stated) within the top-K ranked predictions the deployed head emits. It is read at any budget by offline truncation of a single depth-1000 cache, which is bit-identical to re-running with that budget because top-K selection is order-preserving and never enters the training loss.
- The nominal budget is `K_nom = 300`; the effective budget `K_eff` is the number of distinct true positives the kept slots carry, which falls below `K_nom` when duplicates and false positives occupy slots.
- Evaluation is deploy-faithful: ignore regions are exempted after truncation, not before; matching is greedy in descending confidence at the deployed operating thresholds.
- Primary VisDrone and SKU-110K numbers are reported over two seeds; the DOTA second-domain and YOLOv10 second-detector results are single-run generalization checks.
- Every path is relative to `DART_ROOT` (env var; defaults to the repo root). No absolute paths are hard-coded.

## Citation

Under review. BibTeX will be added on acceptance:

```bibtex
@article{dart_rtil,
  title   = {DART: A Density-Stratified Diagnosis and Training-Free Budget
             Repair of Rank-Truncation Information Loss in Fixed-Budget
             NMS-free Detection},
  author  = {TODO},
  year    = {2026},
  note    = {Under review}
}
```
