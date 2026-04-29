# NAC TRG Response Network

This directory contains an Experiment 5 implementation for pre-NAC TRG response prediction.

The old `coca_trg` package remains the COCA-style segmentation-classification baseline. This package is a separate mask-aware TRG response model.

## Model

```text
pre-NAC CT crop
tumor mask
peritumor ring
ROI statistics
  -> 3D encoder
  -> global pooling + tumor masked pooling + peritumor masked pooling
  -> ROI statistics MLP
  -> binary TRG 0/1 vs 2/3 head
  -> ordinal TRG auxiliary head
```

The mask is not treated as the TRG label. It is used as an ROI prior for tumor-centered crop, tumor pooling, peritumor pooling, and optional input channels.

By default, rows with missing or shape-mismatched masks are skipped because this response model depends on valid tumor ROI. Use `--allow-missing-mask` only for debugging.

## Dry Run

```powershell
& 'C:\Users\msi\.conda\envs\medicine\python.exe' -m nac_trg.train --data-root 'H:\COCA\CMS_npy_simple' --label-xlsx 'H:\COCA\label.xlsx' --output-dir 'H:\COCA\NAC_TRG\outputs\response_probe' --cv-folds 5 --dry-run
```

## 5-Fold Training

```powershell
& 'C:\Users\msi\.conda\envs\medicine\python.exe' -m nac_trg.train --data-root 'H:\COCA\CMS_npy_simple' --label-xlsx 'H:\COCA\label.xlsx' --output-dir 'H:\COCA\NAC_TRG\outputs\response_5fold' --cv-folds 5 --epochs 60 --batch-size 1 --num-workers 2 --target-shape 64,128,128 --train-crop-shape 96,192,192 --tumor-centered-crop-prob 1.0 --ring-radius 7 --balanced-sampler --augment --flip-prob 0.5 --intensity-jitter 0.08 --noise-std 0.02 --base-channels 16 --depth 3 --hidden-dim 64 --dropout 0.3 --lambda-ordinal 0.3 --selection-metric auroc --lr 3e-5 --device cuda
```

Add `--mask-as-input` for the stronger ablation where image, tumor mask, and peritumor ring are passed as 3 input channels.

## Outputs

Each fold writes:

```text
fold_N/best.pt
fold_N/metrics.csv
fold_N/best_val_predictions.csv
```

The root output directory writes:

```text
manifest.csv
cv_summary.csv
```

Primary metrics:

```text
val_auroc
val_pr_auc
val_best_threshold_balanced_accuracy
val_ordinal_mae
```
