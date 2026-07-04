# WP1 full pilot driver: 3 models x 2 datasets -> infer, eval, plot.
# Run from the repo root. Uses the active Python environment.
$ErrorActionPreference = "Stop"
$PY = "python"
$ROOT = Split-Path $PSScriptRoot -Parent
$W = "$ROOT\weights"
$DATA = "$ROOT\data"
$OUT = "$ROOT\experiments\wp1_pilot"

$models = @("yolo26n", "yolo26s", "yolo26m")

foreach ($m in $models) {
    $preds = "$OUT\preds_visdrone_$m.jsonl"
    if (-not (Test-Path $preds)) {
        & $PY "$ROOT\scripts\wp1_infer.py" --model "$W\$m.pt" `
            --images "$DATA\VisDrone2019-DET-val\images" `
            --out $preds --imgsz 1280 --max-det 1000 --device 0 --batch 4
    }
    & $PY "$ROOT\scripts\wp1_eval.py" --dataset visdrone `
        --gt "$DATA\VisDrone2019-DET-val\annotations" `
        --preds $preds --iou 0.5 --out-prefix "$OUT\visdrone_$m"
}

foreach ($m in $models) {
    $preds = "$OUT\preds_crowdhuman_$m.jsonl"
    if (-not (Test-Path $preds)) {
        & $PY "$ROOT\scripts\wp1_infer.py" --model "$W\$m.pt" `
            --images "$DATA\crowdhuman\Images" `
            --out $preds --imgsz 1280 --max-det 1000 --device 0 --batch 4
    }
    & $PY "$ROOT\scripts\wp1_eval.py" --dataset crowdhuman `
        --gt "$DATA\crowdhuman\annotation_val.odgt" `
        --preds $preds --iou 0.5 --out-prefix "$OUT\crowdhuman_$m"
}

# plots: one figure set per dataset comparing models
& $PY "$ROOT\scripts\wp1_plot.py" `
    --per-image "$OUT\visdrone_yolo26n_per_image.csv" "$OUT\visdrone_yolo26s_per_image.csv" "$OUT\visdrone_yolo26m_per_image.csv" `
    --labels "yolo26n" "yolo26s" "yolo26m" `
    --out-dir "$OUT\plots_visdrone" --title "VisDrone-val (COCO weights, imgsz1280)"

& $PY "$ROOT\scripts\wp1_plot.py" `
    --per-image "$OUT\crowdhuman_yolo26n_per_image.csv" "$OUT\crowdhuman_yolo26s_per_image.csv" "$OUT\crowdhuman_yolo26m_per_image.csv" `
    --labels "yolo26n" "yolo26s" "yolo26m" `
    --out-dir "$OUT\plots_crowdhuman" --title "CrowdHuman-val (COCO weights, imgsz1280)"

Write-Output "WP1 pipeline complete."
