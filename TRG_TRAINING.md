# TRG Binary Training

This workspace now contains a COCA-inspired PyTorch pipeline for TRG binary classification.

## Environment

Use the `medicine` conda environment:

```powershell
& 'C:\Users\msi\.conda\envs\medicine\python.exe' -m coca_trg.train --help
```

## Default Task

The default binary rule is:

```text
positive: TRG = 0 or 1
negative: TRG = 2 or 3
```

Use `--binary-rule trg0_vs_123` if the task should instead be:

```text
positive: TRG = 0
negative: TRG = 1, 2, 3
```

## Dry Run

Build the manifest and verify the train/validation split without loading full CT volumes:

```powershell
& 'C:\Users\msi\.conda\envs\medicine\python.exe' -m coca_trg.train `
  --data-root 'H:\COCA\CMS_npy_simple' `
  --label-xlsx 'H:\COCA\label.xlsx' `
  --output-dir 'H:\COCA\outputs\trg' `
  --dry-run
```

Expected current manifest with the default `TRG0/1 vs TRG2/3` rule:

```text
rows: 132
positive TRG0-1: 23
negative TRG2-3: 109
mask shape matches: 127
mask shape mismatches: 5
```

## Build Cache

The original `.npy` files are large `float64` CT volumes. Build a resized cache before longer training so the GPU does not wait on repeated full-volume CPU preprocessing:

```powershell
& 'C:\Users\msi\.conda\envs\medicine\python.exe' -m coca_trg.train `
  --data-root 'H:\COCA\CMS_npy_simple' `
  --label-xlsx 'H:\COCA\label.xlsx' `
  --output-dir 'H:\COCA\outputs\trg_cache_probe' `
  --cache-dir 'H:\COCA\CMS_npy_cache' `
  --target-shape 64,128,128 `
  --prepare-cache-only
```

Progress bars are enabled by default. Add `--no-progress` if redirecting logs to a file.

## Train Classification

Start with cached inputs, batch size 1, and two DataLoader workers:

```powershell
& 'C:\Users\msi\.conda\envs\medicine\python.exe' -m coca_trg.train `
  --data-root 'H:\COCA\CMS_npy_simple' `
  --label-xlsx 'H:\COCA\label.xlsx' `
  --output-dir 'H:\COCA\outputs\trg' `
  --cache-dir 'H:\COCA\CMS_npy_cache' `
  --epochs 50 `
  --batch-size 1 `
  --num-workers 2 `
  --target-shape 64,128,128 `
  --base-channels 8 `
  --depth 3 `
  --device cuda
```

The best checkpoint is saved to:

```text
H:\COCA\outputs\trg\best.pt
```

## Optional Mask Auxiliary Loss

The code can add tumor mask segmentation loss with `--lambda-seg`. Cached training supports masks: image caches are saved as float16, mask caches are saved as uint8, and the five current image/mask shape mismatches are automatically ignored for the segmentation loss.

```powershell
& 'C:\Users\msi\.conda\envs\medicine\python.exe' -m coca_trg.train `
  --data-root 'H:\COCA\CMS_npy_simple' `
  --label-xlsx 'H:\COCA\label.xlsx' `
  --output-dir 'H:\COCA\outputs\trg_seg_aux' `
  --cache-dir 'H:\COCA\CMS_npy_cache' `
  --epochs 50 `
  --batch-size 1 `
  --num-workers 2 `
  --target-shape 64,128,128 `
  --base-channels 8 `
  --depth 3 `
  --lambda-seg 0.2 `
  --device cuda
```

The classifier fuses decoder pooled features with four segmentation-derived features by default:

```text
mean tumor probability
max tumor probability
tumor probability standard deviation
hard tumor volume fraction at 0.5 threshold
```

Add `--no-fusion-features` to disable this branch for ablation.

## Optional Tumor-Centered Training Crops

For uncached training, positive cases can be cropped around tumor-mask voxels before resizing to `--target-shape`. This uses the true mask only for training-time sampling and segmentation loss; the mask is still not passed into the model input.

```powershell
& 'C:\Users\msi\.conda\envs\medicine\python.exe' -m coca_trg.train `
  --data-root 'H:\COCA\CMS_npy_simple' `
  --label-xlsx 'H:\COCA\label.xlsx' `
  --output-dir 'H:\COCA\outputs\trg_seg_aux_tumor_crop' `
  --epochs 50 `
  --batch-size 1 `
  --num-workers 2 `
  --target-shape 64,128,128 `
  --train-crop-shape 96,192,192 `
  --tumor-centered-crop-prob 0.8 `
  --base-channels 8 `
  --depth 3 `
  --lambda-seg 0.2 `
  --device cuda
```

`--train-crop-shape` cannot be combined with `--cache-dir`, because the cached data has already been resized.

## Stratified 5-Fold CV With Class Balancing

Use `--cv-folds 5` to train five stratified folds. Each fold writes its own `fold_N` directory with `best.pt`, `metrics.csv`, and `best_val_predictions.csv`; the root output directory also gets `cv_summary.csv`.

Each epoch now records threshold-dependent metrics and threshold-free ranking metrics:

```text
val_balanced_accuracy              fixed 0.5 threshold
val_best_threshold_balanced_accuracy
val_auroc                          ranking ability across all thresholds
val_pr_auc                         positive-class precision-recall ranking
```

`cv_summary.csv` includes the selected best epoch per fold, plus `best_epoch_auroc` and `best_epoch_pr_auc`.

`--balanced-sampler` oversamples the minority class during training. When enabled, classification `pos_weight` is set to 1.0 to avoid double-correcting the positive class.

Training augmentation is off by default. Enable it only for training with `--augment`; validation is never augmented.

```powershell
& 'C:\Users\msi\.conda\envs\medicine\python.exe' -m coca_trg.train `
  --data-root 'H:\COCA\CMS_npy_simple' `
  --label-xlsx 'H:\COCA\label.xlsx' `
  --output-dir 'H:\COCA\outputs\trg_5fold_balanced_aug' `
  --cv-folds 5 `
  --epochs 60 `
  --batch-size 1 `
  --num-workers 2 `
  --target-shape 64,128,128 `
  --train-crop-shape 96,192,192 `
  --tumor-centered-crop-prob 0.8 `
  --balanced-sampler `
  --augment `
  --flip-prob 0.5 `
  --intensity-jitter 0.08 `
  --noise-std 0.02 `
  --base-channels 16 `
  --depth 3 `
  --lambda-seg 0.2 `
  --lr 3e-5 `
  --device cuda
```

For a fast split check without training:

```powershell
& 'C:\Users\msi\.conda\envs\medicine\python.exe' -m coca_trg.train `
  --data-root 'H:\COCA\CMS_npy_simple' `
  --label-xlsx 'H:\COCA\label.xlsx' `
  --output-dir 'H:\COCA\outputs\trg_5fold_probe' `
  --cv-folds 5 `
  --dry-run
```
