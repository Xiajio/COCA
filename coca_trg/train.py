from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import random
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from .cache import CachedTRGVolumeDataset, prepare_cache
from .dataset import TRGVolumeDataset
from .losses import segmentation_loss
from .metadata import ManifestRow, build_manifest, manifest_summary
from .models import COCAForTRG


def _parse_shape(text: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in text.split(",")]
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError("shape must be formatted as D,H,W, e.g. 64,128,128")
    return tuple(parts)


def _stratified_split(
    rows: Sequence[ManifestRow],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[ManifestRow], list[ManifestRow]]:
    rng = random.Random(seed)
    positives = [row for row in rows if row.binary_label == 1]
    negatives = [row for row in rows if row.binary_label == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)

    def take_val(group: list[ManifestRow]) -> tuple[list[ManifestRow], list[ManifestRow]]:
        if len(group) <= 1:
            return group, []
        val_count = max(1, round(len(group) * val_fraction))
        val_count = min(val_count, len(group) - 1)
        return group[val_count:], group[:val_count]

    train_pos, val_pos = take_val(positives)
    train_neg, val_neg = take_val(negatives)
    train_rows = train_pos + train_neg
    val_rows = val_pos + val_neg
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def _stratified_kfold_splits(
    rows: Sequence[ManifestRow],
    *,
    n_splits: int,
    seed: int,
) -> list[tuple[list[ManifestRow], list[ManifestRow]]]:
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    rng = random.Random(seed)
    positives = [row for row in rows if row.binary_label == 1]
    negatives = [row for row in rows if row.binary_label == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)

    fold_rows: list[list[ManifestRow]] = [[] for _ in range(n_splits)]
    for group in (positives, negatives):
        for index, row in enumerate(group):
            fold_rows[index % n_splits].append(row)

    all_rows = list(rows)
    splits = []
    for fold_index, val_rows in enumerate(fold_rows):
        val_ids = {row.case_id for row in val_rows}
        train_rows = [row for row in all_rows if row.case_id not in val_ids]
        rng.shuffle(train_rows)
        rng.shuffle(val_rows)
        splits.append((train_rows, val_rows))
    return splits


def _balanced_sample_weights(rows: Sequence[ManifestRow]) -> torch.Tensor:
    positives = sum(row.binary_label == 1 for row in rows)
    negatives = sum(row.binary_label == 0 for row in rows)
    if positives == 0 or negatives == 0:
        raise ValueError("balanced sampling requires both positive and negative rows")
    return torch.tensor(
        [1.0 / positives if row.binary_label == 1 else 1.0 / negatives for row in rows],
        dtype=torch.double,
    )


def _make_balanced_sampler(rows: Sequence[ManifestRow], *, seed: int) -> WeightedRandomSampler:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        _balanced_sample_weights(rows),
        num_samples=len(rows),
        replacement=True,
        generator=generator,
    )


def _write_manifest_csv(rows: Sequence[ManifestRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = asdict(row)
            record["image_path"] = str(row.image_path)
            record["mask_path"] = str(row.mask_path) if row.mask_path is not None else ""
            writer.writerow(record)


def _classification_metrics(
    probs_or_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    threshold: float = 0.5,
    input_is_probability: bool = False,
) -> dict[str, float]:
    probs = probs_or_logits.float() if input_is_probability else torch.sigmoid(probs_or_logits.float())
    preds = (probs >= threshold).float()
    labels = labels.float()
    accuracy = (preds == labels).float().mean().item()
    pos_mask = labels == 1
    neg_mask = labels == 0
    sensitivity = (preds[pos_mask] == 1).float().mean().item() if pos_mask.any() else 0.0
    specificity = (preds[neg_mask] == 0).float().mean().item() if neg_mask.any() else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
    }


def _ranking_metrics(probs: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probs = probs.detach().cpu().float()
    labels = labels.detach().cpu().float()
    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_count = int(pos_mask.sum().item())
    neg_count = int(neg_mask.sum().item())
    if pos_count == 0 or neg_count == 0:
        return {"auroc": float("nan"), "pr_auc": float("nan")}

    pos_probs = probs[pos_mask]
    neg_probs = probs[neg_mask]
    comparisons = (pos_probs[:, None] > neg_probs[None, :]).float()
    ties = (pos_probs[:, None] == neg_probs[None, :]).float() * 0.5
    auroc = (comparisons + ties).mean().item()

    order = torch.argsort(probs, descending=True)
    sorted_labels = labels[order]
    tp_cumulative = torch.cumsum(sorted_labels, dim=0)
    ranks = torch.arange(1, len(sorted_labels) + 1, dtype=torch.float32)
    precision_at_rank = tp_cumulative / ranks
    pr_auc = precision_at_rank[sorted_labels == 1].mean().item()
    return {"auroc": auroc, "pr_auc": pr_auc}


def _search_best_threshold(probs: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probs = probs.detach().cpu().float()
    labels = labels.detach().cpu().float()
    candidates = sorted({round(float(value), 6) for value in probs.tolist()} | {0.0, 0.5, 1.0})
    best: dict[str, float] | None = None
    for threshold in candidates:
        metrics = _classification_metrics(
            probs,
            labels,
            threshold=threshold,
            input_is_probability=True,
        )
        if best is None:
            best = metrics
            continue
        better_balanced_accuracy = metrics["balanced_accuracy"] > best["balanced_accuracy"]
        same_balanced_accuracy = metrics["balanced_accuracy"] == best["balanced_accuracy"]
        closer_to_default = abs(metrics["threshold"] - 0.5) < abs(best["threshold"] - 0.5)
        if better_balanced_accuracy or (same_balanced_accuracy and closer_to_default):
            best = metrics
    assert best is not None
    return best


def _make_prediction_rows(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    trgs: torch.Tensor,
    case_ids: Sequence[str],
) -> list[dict[str, object]]:
    probs = torch.sigmoid(logits.detach().cpu()).tolist()
    label_values = labels.detach().cpu().long().tolist()
    trg_values = trgs.detach().cpu().long().tolist()
    rows: list[dict[str, object]] = []
    for case_id, prob, label, trg in zip(case_ids, probs, label_values, trg_values):
        rows.append(
            {
                "case_id": str(case_id),
                "probability": round(float(prob), 6),
                "label": int(label),
                "trg": int(trg),
            }
        )
    return rows


def _write_prediction_csv(rows: Sequence[dict[str, object]], path: Path, *, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case_id", "probability", "prediction", "label", "trg"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            probability = float(row["probability"])
            record = dict(row)
            record["prediction"] = int(probability >= threshold)
            writer.writerow(record)


def _run_epoch(
    *,
    model: COCAForTRG,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    pos_weight: torch.Tensor,
    lambda_seg: float,
    progress: bool,
    desc: str,
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_cls = 0.0
    total_seg = 0.0
    all_logits = []
    all_labels = []
    prediction_rows: list[dict[str, object]] = []

    iterator = tqdm(loader, desc=desc, unit="batch", leave=False, disable=not progress)
    for batch in iterator:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        outputs = model(images)
        cls_loss = F.binary_cross_entropy_with_logits(
            outputs["cls_logit"],
            labels,
            pos_weight=pos_weight,
        )
        seg_loss = torch.zeros((), device=device)
        if lambda_seg > 0 and "mask" in batch:
            has_mask = batch["has_mask"].to(device, non_blocking=True)
            if has_mask.any():
                masks = batch["mask"].to(device, non_blocking=True)
                seg_loss = segmentation_loss(outputs["seg_logits"][has_mask], masks[has_mask])
        loss = cls_loss + lambda_seg * seg_loss

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = int(images.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_cls += float(cls_loss.detach().cpu()) * batch_size
        total_seg += float(seg_loss.detach().cpu()) * batch_size
        all_logits.append(outputs["cls_logit"].detach().cpu())
        all_labels.append(labels.detach().cpu())
        trgs = batch["trg"].detach().cpu()
        prediction_rows.extend(
            _make_prediction_rows(
                logits=outputs["cls_logit"],
                labels=labels,
                trgs=batch["trg"],
                case_ids=batch["case_id"],
            )
        )
        seen = len(all_labels)
        iterator.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}", batches=seen)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = torch.sigmoid(logits)
    metrics = _classification_metrics(logits, labels)
    metrics.update(_ranking_metrics(probs, labels))
    best_threshold_metrics = _search_best_threshold(probs, labels)
    count = len(loader.dataset)
    metrics.update(
        {
            "loss": total_loss / count,
            "cls_loss": total_cls / count,
            "seg_loss": total_seg / count,
            "best_threshold": best_threshold_metrics["threshold"],
            "best_threshold_accuracy": best_threshold_metrics["accuracy"],
            "best_threshold_sensitivity": best_threshold_metrics["sensitivity"],
            "best_threshold_specificity": best_threshold_metrics["specificity"],
            "best_threshold_balanced_accuracy": best_threshold_metrics["balanced_accuracy"],
        }
    )
    metrics["prediction_rows"] = prediction_rows
    return metrics


def _write_metrics_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "train_cls_loss",
        "train_seg_loss",
        "train_auroc",
        "train_pr_auc",
        "val_loss",
        "val_cls_loss",
        "val_seg_loss",
        "val_auroc",
        "val_pr_auc",
        "val_accuracy",
        "val_sensitivity",
        "val_specificity",
        "val_balanced_accuracy",
        "val_best_threshold",
        "val_best_threshold_accuracy",
        "val_best_threshold_sensitivity",
        "val_best_threshold_specificity",
        "val_best_threshold_balanced_accuracy",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_cv_summary(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "fold",
        "train_rows",
        "train_positive",
        "train_negative",
        "val_rows",
        "val_positive",
        "val_negative",
        "best_epoch",
        "best_threshold",
        "best_threshold_accuracy",
        "best_threshold_sensitivity",
        "best_threshold_specificity",
        "best_threshold_balanced_accuracy",
        "best_epoch_auroc",
        "best_epoch_pr_auc",
        "fixed_threshold_balanced_accuracy",
        "best_checkpoint",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _metric_payload(metrics: dict[str, object]) -> dict[str, float]:
    return {key: value for key, value in metrics.items() if key != "prediction_rows"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a COCA-inspired model for TRG binary classification.")
    parser.add_argument("--data-root", type=Path, default=Path("CMS_npy_simple"))
    parser.add_argument("--label-xlsx", type=Path, default=Path("label.xlsx"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "trg")
    parser.add_argument("--binary-rule", choices=["trg0_vs_123", "trg01_vs_23"], default="trg01_vs_23")
    parser.add_argument("--cv-folds", type=int, default=1, help="Run stratified K-fold CV when > 1.")
    parser.add_argument("--target-shape", type=_parse_shape, default=(64, 128, 128))
    parser.add_argument("--train-crop-shape", type=_parse_shape, default=None, help="Optional original-space D,H,W crop for training only.")
    parser.add_argument("--tumor-centered-crop-prob", type=float, default=0.0, help="Probability of centering a positive training crop on a tumor-mask voxel.")
    parser.add_argument("--window", type=float, nargs=2, default=(-150.0, 250.0))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.set_defaults(fusion_features=True)
    parser.add_argument("--no-fusion-features", dest="fusion_features", action="store_false", help="Disable tumor probability/volume feature fusion.")
    parser.add_argument("--lambda-seg", type=float, default=0.0)
    parser.add_argument("--balanced-sampler", action="store_true", help="Use inverse-frequency weighted sampling for training batches.")
    parser.add_argument("--augment", action="store_true", help="Enable training-only image/mask augmentation.")
    parser.add_argument("--flip-prob", type=float, default=0.0, help="Per-axis H/W flip probability when --augment is enabled.")
    parser.add_argument("--intensity-jitter", type=float, default=0.0, help="Random intensity scale/shift magnitude when --augment is enabled.")
    parser.add_argument("--noise-std", type=float, default=0.0, help="Gaussian noise std in normalized image units when --augment is enabled.")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit rows for smoke tests; 0 uses all rows.")
    parser.add_argument("--dry-run", action="store_true", help="Build manifest and print split summary without training.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Directory for preprocessed resized CT cache.")
    parser.add_argument("--cache-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--prepare-cache-only", action="store_true", help="Build cache and exit without training.")
    parser.set_defaults(progress=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false", help="Disable tqdm progress bars.")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.cv_folds < 1:
        raise SystemExit("--cv-folds must be >= 1.")
    if args.prepare_cache_only and args.cache_dir is None:
        raise SystemExit("--prepare-cache-only requires --cache-dir.")
    if not 0.0 <= args.tumor_centered_crop_prob <= 1.0:
        raise SystemExit("--tumor-centered-crop-prob must be between 0 and 1.")
    if not 0.0 <= args.flip_prob <= 1.0:
        raise SystemExit("--flip-prob must be between 0 and 1.")
    if args.intensity_jitter < 0:
        raise SystemExit("--intensity-jitter must be >= 0.")
    if args.noise_std < 0:
        raise SystemExit("--noise-std must be >= 0.")
    if args.train_crop_shape is not None and args.cache_dir is not None:
        raise SystemExit("--train-crop-shape uses original-space arrays and cannot be combined with --cache-dir.")


def _train_one_split(
    args: argparse.Namespace,
    *,
    train_rows: Sequence[ManifestRow],
    val_rows: Sequence[ManifestRow],
    output_dir: Path,
    manifest_summary_payload: dict[str, int],
    fold_index: int | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest_csv(train_rows, output_dir / "train_manifest.csv")
    _write_manifest_csv(val_rows, output_dir / "val_manifest.csv")

    include_masks = args.lambda_seg > 0
    split_seed = args.seed if fold_index is None else args.seed + fold_index
    if args.cache_dir is not None:
        train_dataset = CachedTRGVolumeDataset(
            train_rows,
            cache_dir=args.cache_dir,
            target_shape=args.target_shape,
            window=tuple(args.window),
            dtype=args.cache_dtype,
            include_masks=include_masks,
            augment=args.augment,
            flip_prob=args.flip_prob,
            intensity_jitter=args.intensity_jitter,
            noise_std=args.noise_std,
            seed=split_seed,
        )
        val_dataset = CachedTRGVolumeDataset(
            val_rows,
            cache_dir=args.cache_dir,
            target_shape=args.target_shape,
            window=tuple(args.window),
            dtype=args.cache_dtype,
            include_masks=include_masks,
        )
    else:
        train_dataset = TRGVolumeDataset(
            train_rows,
            target_shape=args.target_shape,
            window=tuple(args.window),
            include_masks=include_masks,
            crop_shape=args.train_crop_shape,
            tumor_centered_crop_prob=args.tumor_centered_crop_prob,
            augment=args.augment,
            flip_prob=args.flip_prob,
            intensity_jitter=args.intensity_jitter,
            noise_std=args.noise_std,
            seed=split_seed,
        )
        val_dataset = TRGVolumeDataset(
            val_rows,
            target_shape=args.target_shape,
            window=tuple(args.window),
            include_masks=include_masks,
        )

    sampler = _make_balanced_sampler(train_rows, seed=split_seed) if args.balanced_sampler else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    device = torch.device(args.device)
    model = COCAForTRG(
        base_channels=args.base_channels,
        depth=args.depth,
        dropout=args.dropout,
        fusion_features=args.fusion_features,
    ).to(device)
    train_pos = sum(row.binary_label == 1 for row in train_rows)
    train_neg = sum(row.binary_label == 0 for row in train_rows)
    pos_weight_value = 1.0 if args.balanced_sampler else train_neg / max(train_pos, 1)
    pos_weight = torch.tensor([pos_weight_value], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_balanced_accuracy = -1.0
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    best_path = output_dir / "best.pt"
    metrics_csv = output_dir / "metrics.csv"
    if metrics_csv.exists():
        metrics_csv.unlink()
    fold_label = f"fold={fold_index + 1} " if fold_index is not None else ""
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            pos_weight=pos_weight,
            lambda_seg=args.lambda_seg,
            progress=args.progress,
            desc=f"{fold_label}epoch {epoch:03d}/{args.epochs} train",
        )
        train_metrics.pop("prediction_rows")
        with torch.no_grad():
            val_metrics = _run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                optimizer=None,
                pos_weight=pos_weight,
                lambda_seg=args.lambda_seg,
                progress=args.progress,
                desc=f"{fold_label}epoch {epoch:03d}/{args.epochs} val",
            )
        val_prediction_rows = val_metrics.pop("prediction_rows")
        val_threshold = float(val_metrics["best_threshold"])
        _write_prediction_csv(
            val_prediction_rows,
            output_dir / f"val_predictions_epoch_{epoch:03d}.csv",
            threshold=val_threshold,
        )
        _write_metrics_row(
            metrics_csv,
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_cls_loss": train_metrics["cls_loss"],
                "train_seg_loss": train_metrics["seg_loss"],
                "train_auroc": train_metrics["auroc"],
                "train_pr_auc": train_metrics["pr_auc"],
                "val_loss": val_metrics["loss"],
                "val_cls_loss": val_metrics["cls_loss"],
                "val_seg_loss": val_metrics["seg_loss"],
                "val_auroc": val_metrics["auroc"],
                "val_pr_auc": val_metrics["pr_auc"],
                "val_accuracy": val_metrics["accuracy"],
                "val_sensitivity": val_metrics["sensitivity"],
                "val_specificity": val_metrics["specificity"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_best_threshold": val_metrics["best_threshold"],
                "val_best_threshold_accuracy": val_metrics["best_threshold_accuracy"],
                "val_best_threshold_sensitivity": val_metrics["best_threshold_sensitivity"],
                "val_best_threshold_specificity": val_metrics["best_threshold_specificity"],
                "val_best_threshold_balanced_accuracy": val_metrics["best_threshold_balanced_accuracy"],
            },
        )
        print(
            f"{fold_label}"
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} "
            f"val_auroc={val_metrics['auroc']:.4f} "
            f"val_pr_auc={val_metrics['pr_auc']:.4f} "
            f"val_best_thr={val_metrics['best_threshold']:.4f} "
            f"val_best_bal_acc={val_metrics['best_threshold_balanced_accuracy']:.4f} "
            f"val_sens={val_metrics['sensitivity']:.4f} "
            f"val_spec={val_metrics['specificity']:.4f}"
        )
        if val_metrics["best_threshold_balanced_accuracy"] > best_balanced_accuracy:
            best_balanced_accuracy = float(val_metrics["best_threshold_balanced_accuracy"])
            best_epoch = epoch
            best_metrics = _metric_payload(val_metrics)
            _write_prediction_csv(
                val_prediction_rows,
                output_dir / "best_val_predictions.csv",
                threshold=val_threshold,
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "fold": fold_index + 1 if fold_index is not None else None,
                    "val_metrics": best_metrics,
                    "best_threshold": val_metrics["best_threshold"],
                    "best_val_predictions_csv": str(output_dir / "best_val_predictions.csv"),
                    "manifest_summary": manifest_summary_payload,
                },
                best_path,
            )
    print(f"{fold_label}Best checkpoint: {best_path} balanced_accuracy={best_balanced_accuracy:.4f}")
    return {
        "fold": fold_index + 1 if fold_index is not None else 0,
        "train_rows": len(train_rows),
        "train_positive": train_pos,
        "train_negative": train_neg,
        "val_rows": len(val_rows),
        "val_positive": sum(row.binary_label == 1 for row in val_rows),
        "val_negative": sum(row.binary_label == 0 for row in val_rows),
        "best_epoch": best_epoch,
        "best_threshold": best_metrics.get("best_threshold", ""),
        "best_threshold_accuracy": best_metrics.get("best_threshold_accuracy", ""),
        "best_threshold_sensitivity": best_metrics.get("best_threshold_sensitivity", ""),
        "best_threshold_specificity": best_metrics.get("best_threshold_specificity", ""),
        "best_threshold_balanced_accuracy": best_metrics.get("best_threshold_balanced_accuracy", ""),
        "best_epoch_auroc": best_metrics.get("auroc", ""),
        "best_epoch_pr_auc": best_metrics.get("pr_auc", ""),
        "fixed_threshold_balanced_accuracy": best_metrics.get("balanced_accuracy", ""),
        "best_checkpoint": str(best_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _validate_args(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    rows = build_manifest(args.data_root, args.label_xlsx, binary_rule=args.binary_rule)
    if args.max_samples:
        rows = rows[: args.max_samples]
    if len(rows) < 4:
        raise SystemExit("Need at least 4 labeled rows to train and validate.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest_csv(rows, args.output_dir / "manifest.csv")
    summary = manifest_summary(rows)
    print(f"Manifest: {summary}")

    include_masks = args.lambda_seg > 0
    if args.cv_folds > 1:
        positives = sum(row.binary_label == 1 for row in rows)
        negatives = sum(row.binary_label == 0 for row in rows)
        if args.cv_folds > positives or args.cv_folds > negatives:
            raise SystemExit(
                "--cv-folds cannot exceed the positive or negative case count "
                f"(positive={positives}, negative={negatives})."
            )
        folds = _stratified_kfold_splits(rows, n_splits=args.cv_folds, seed=args.seed)
        for fold_index, (train_rows, val_rows) in enumerate(folds, start=1):
            print(
                f"Fold {fold_index}/{args.cv_folds}: "
                f"train={manifest_summary(train_rows)}, val={manifest_summary(val_rows)}"
            )
        if args.dry_run:
            return 0

        if args.cache_dir is not None:
            prepare_cache(
                rows,
                cache_dir=args.cache_dir,
                target_shape=args.target_shape,
                window=tuple(args.window),
                dtype=args.cache_dtype,
                rebuild=args.rebuild_cache,
                progress=args.progress,
                include_masks=include_masks,
            )
            if args.prepare_cache_only:
                return 0

        cv_rows = []
        for fold_index, (train_rows, val_rows) in enumerate(folds):
            cv_rows.append(
                _train_one_split(
                    args,
                    train_rows=train_rows,
                    val_rows=val_rows,
                    output_dir=args.output_dir / f"fold_{fold_index + 1}",
                    manifest_summary_payload=summary,
                    fold_index=fold_index,
                )
            )
        _write_cv_summary(args.output_dir / "cv_summary.csv", cv_rows)
        best_balanced = [
            float(row["best_threshold_balanced_accuracy"])
            for row in cv_rows
            if row["best_threshold_balanced_accuracy"] != ""
        ]
        if best_balanced:
            values = torch.tensor(best_balanced)
            std = values.std(unbiased=False).item()
            aurocs = torch.tensor(
                [float(row["best_epoch_auroc"]) for row in cv_rows if row["best_epoch_auroc"] != ""],
                dtype=torch.float32,
            )
            pr_aucs = torch.tensor(
                [float(row["best_epoch_pr_auc"]) for row in cv_rows if row["best_epoch_pr_auc"] != ""],
                dtype=torch.float32,
            )
            print(
                f"CV summary: folds={len(best_balanced)} "
                f"mean_best_bal_acc={values.mean().item():.4f} "
                f"std_best_bal_acc={std:.4f} "
                f"mean_auroc={aurocs.mean().item():.4f} "
                f"mean_pr_auc={pr_aucs.mean().item():.4f}"
            )
        return 0

    train_rows, val_rows = _stratified_split(rows, val_fraction=args.val_fraction, seed=args.seed)
    print(f"Split: train={manifest_summary(train_rows)}, val={manifest_summary(val_rows)}")
    if args.dry_run:
        return 0

    if args.cache_dir is not None:
        prepare_cache(
            rows,
            cache_dir=args.cache_dir,
            target_shape=args.target_shape,
            window=tuple(args.window),
            dtype=args.cache_dtype,
            rebuild=args.rebuild_cache,
            progress=args.progress,
            include_masks=include_masks,
        )
        if args.prepare_cache_only:
            return 0
    _train_one_split(
        args,
        train_rows=train_rows,
        val_rows=val_rows,
        output_dir=args.output_dir,
        manifest_summary_payload=summary,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
