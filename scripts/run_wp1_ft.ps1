# Re-analysis driver: run the full WP1 analysis suite on finetuned weights.
# Usage: .\run_wp1_ft.ps1 -Weights <path\to\best.pt> -Tag ft_n -Device 0
param(
    [Parameter(Mandatory = $true)][string]$Weights,
    [Parameter(Mandatory = $true)][string]$Tag,
    [string]$Device = "0"
)
$ErrorActionPreference = "Stop"
$PY = "python"
$ROOT = Split-Path $PSScriptRoot -Parent
$OUT = "$ROOT\experiments\wp1_ft"
New-Item -ItemType Directory -Force -Path $OUT | Out-Null
Set-Location "$ROOT\scripts"

$preds = "$OUT\preds_visdrone_$Tag.jsonl"
& $PY "$ROOT\scripts\wp1_infer.py" --model $Weights `
    --images "$ROOT\data\VisDrone2019-DET-val\images" `
    --out $preds --imgsz 1280 --max-det 1000 --device $Device
& $PY "$ROOT\scripts\wp1_eval.py" --dataset visdrone `
    --gt "$ROOT\data\VisDrone2019-DET-val\annotations" `
    --preds $preds --iou 0.5 --out-prefix "$OUT\visdrone_$Tag"
& $PY "$ROOT\scripts\wp1_eval.py" --dataset visdrone `
    --gt "$ROOT\data\VisDrone2019-DET-val\annotations" `
    --preds $preds --iou 0.75 --out-prefix "$OUT\visdrone_${Tag}_iou75"
& $PY "$ROOT\scripts\wp1_fppi.py" --dataset visdrone `
    --gt "$ROOT\data\VisDrone2019-DET-val\annotations" --preds $preds
& $PY "$ROOT\scripts\wp1_local_density.py" --dataset visdrone `
    --gt "$ROOT\data\VisDrone2019-DET-val\annotations" `
    --preds $preds --out-prefix "$OUT\visdrone_${Tag}_local"

# o2m path on same finetuned weights (H3 control, in-domain)
$predsO = "$OUT\preds_visdrone_${Tag}_o2m.jsonl"
& $PY "$ROOT\scripts\wp1_infer.py" --model $Weights `
    --images "$ROOT\data\VisDrone2019-DET-val\images" `
    --out $predsO --imgsz 1280 --max-det 1000 --device $Device --o2m
& $PY "$ROOT\scripts\wp1_eval.py" --dataset visdrone `
    --gt "$ROOT\data\VisDrone2019-DET-val\annotations" `
    --preds $predsO --iou 0.5 --out-prefix "$OUT\visdrone_${Tag}_o2m"

Write-Output "FT analysis complete for $Tag"
