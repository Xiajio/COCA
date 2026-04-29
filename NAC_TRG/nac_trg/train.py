from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import random
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

from .cache import CachedNACResponseDataset, prepare_cache
from .dataset import NACResponseDataset, ROI_STATS_DIM
from .losses import ordinal_prediction, response_loss
from .metadata import ManifestRow, build_manifest, manifest_summary
from .models import TRGResponseNet


def _parse_shape(text: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in text.split(",")]
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError("shape must be formatted as D,H,W")
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

    def split_group(group: list[ManifestRow]) -> tuple[list[ManifestRow], list[ManifestRow]]:
        val_count = max(1, round(len(group) * val_fraction))
        val_count = min(val_count, len(group) - 1)
        return group[val_count:], group[:val_count]

    train_pos, val_pos = split_group(positives)
    train_neg, val_neg = split_group(negatives)
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
    folds: list[list[ManifestRow]] = [[] for _ in range(n_splits)]
    for group in (positives, negatives):
        for index, row in enumerate(group):
            folds[index % n_splits].append(row)
    all_rows = list(rows)
    result = []
    for val_rows in folds:
        val_ids = {row.case_id for row in val_rows}
        train_rows = [row for row in all_rows if row.case_id not in val_ids]
        rng.shuffle(train_rows)
        rng.shuffle(val_rows)
        result.append((train_rows, val_rows))
    return result


def _balanced_sample_weights(rows: Sequence[ManifestRow]) -> torch.Tensor:
    positives = sum(row.binary_label == 1 for row in rows)
    negatives = sum(row.binary_label == 0 for row in rows)
    if positives == 0 or negatives == 0:
        raise ValueError("balanced sampling requires both classes")
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


def _classification_metrics(
    probs_or_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    threshold: float = 0.5,
    input_is_probability: bool = False,
) -> dict[str, float]:
    probs = probs_or_logits.float() if input_is_probability else torch.sigmoid(probs_or_logits.float())
    labels = labels.float()
    preds = (probs >= threshold).float()
    pos_mask = labels == 1
    neg_mask = labels == 0
    sensitivity = (preds[pos_mask] == 1).float().mean().item() if pos_mask.any() else 0.0
    specificity = (preds[neg_mask] == 0).float().mean().item() if neg_mask.any() else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": (preds == labels).float().mean().item(),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
    }


def _ranking_metrics(probs: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probs = probs.detach().cpu().float()
    labels = labels.detach().cpu().float()
    pos = labels == 1
    neg = labels == 0
    if not pos.any() or not neg.any():
        return {"auroc": float("nan"), "pr_auc": float("nan")}
    comparisons = (probs[pos][:, None] > probs[neg][None, :]).float()
    ties = (probs[pos][:, None] == probs[neg][None, :]).float() * 0.5
    auroc = (comparisons + ties).mean().item()
    order = torch.argsort(probs, descending=True)
    sorted_labels = labels[order]
    tp_cum = torch.cumsum(sorted_labels, dim=0)
    ranks = torch.arange(1, len(sorted_labels) + 1, dtype=torch.float32)
    precision = tp_cum / ranks
    pr_auc = precision[sorted_labels == 1].mean().item()
    return {"auroc": auroc, "pr_auc": pr_auc}


def _search_best_threshold(probs: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probs = probs.detach().cpu().float()
    labels = labels.detach().cpu().float()
    candidates = sorted({round(float(value), 6) for value in probs.tolist()} | {0.0, 0.5, 1.0})
    best: dict[str, float] | None = None
    for threshold in candidates:
        metrics = _classification_metrics(probs, labels, threshold=threshold, input_is_probability=True)
        if best is None:
            best = metrics
            continue
        if metrics["balanced_accuracy"] > best["balanced_accuracy"]:
            best = metrics
        elif metrics["balanced_accuracy"] == best["balanced_accuracy"]:
            if abs(metrics["threshold"] - 0.5) < abs(best["threshold"] - 0.5):
                best = metrics
    assert best is not None
    return best


def _write_manifest_csv(rows: Sequence[ManifestRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = asdict(row)
            record["image_path"] = str(row.image_path)
            record["mask_path"] = str(row.mask_path) if row.mask_path else ""
            writer.writerow(record)


def _prediction_rows(
    *,
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    trgs: torch.Tensor,
    case_ids: Sequence[str],
) -> list[dict[str, object]]:
    probs = torch.sigmoid(outputs["binary_logit"].detach().cpu()).tolist()
    ordinal_preds = ordinal_prediction(outputs["ordinal_logits"].detach().cpu()).tolist()
    label_values = labels.detach().cpu().long().tolist()
    trg_values = trgs.detach().cpu().long().tolist()
    rows = []
    for case_id, prob, label, trg, ordinal_pred in zip(case_ids, probs, label_values, trg_values, ordinal_preds):
        rows.append(
            {
                "case_id": str(case_id),
                "probability": round(float(prob), 6),
                "label": int(label),
                "trg": int(trg),
                "ordinal_pred": int(ordinal_pred),
            }
        )
    return rows


def _write_prediction_csv(rows: Sequence[dict[str, object]], path: Path, *, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case_id", "probability", "prediction", "label", "trg", "ordinal_pred"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = dict(row)
            record["prediction"] = int(float(row["probability"]) >= threshold)
            writer.writerow(record)


def _run_epoch(
    *,
    model: TRGResponseNet,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    lambda_ordinal: float,
    pos_weight: torch.Tensor,
    progress: bool,
    desc: str,
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "binary_loss": 0.0, "ordinal_loss": 0.0}
    all_logits = []
    all_labels = []
    all_trgs = []
    prediction_rows: list[dict[str, object]] = []
    iterator = tqdm(loader, desc=desc, unit="batch", leave=False, disable=not progress)
    for batch in iterator:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["tumor_mask"].to(device, non_blocking=True)
        rings = batch["peritumor_ring"].to(device, non_blocking=True)
        stats = batch["roi_stats"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        trgs = batch["trg"].to(device, non_blocking=True)
        outputs = model(image=images, tumor_mask=masks, peritumor_ring=rings, roi_stats=stats)
        losses = response_loss(outputs, labels, trgs, lambda_ordinal=lambda_ordinal, pos_weight=pos_weight)
        if training:
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            optimizer.step()
        batch_size = int(images.shape[0])
        for key in totals:
            totals[key] += float(losses[key].detach().cpu()) * batch_size
        all_logits.append(outputs["binary_logit"].detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_trgs.append(trgs.detach().cpu())
        prediction_rows.extend(_prediction_rows(outputs=outputs, labels=labels, trgs=trgs, case_ids=batch["case_id"]))
        iterator.set_postfix(loss=f"{float(losses['loss'].detach().cpu()):.4f}")

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    trgs = torch.cat(all_trgs)
    probs = torch.sigmoid(logits)
    metrics = _classification_metrics(logits, labels)
    metrics.update(_ranking_metrics(probs, labels))
    best = _search_best_threshold(probs, labels)
    count = len(loader.dataset)
    ordinal_pred = torch.tensor([row["ordinal_pred"] for row in prediction_rows])
    metrics.update(
        {
            "loss": totals["loss"] / count,
            "binary_loss": totals["binary_loss"] / count,
            "ordinal_loss": totals["ordinal_loss"] / count,
            "ordinal_mae": (ordinal_pred.float() - trgs.float()).abs().mean().item(),
            "best_threshold": best["threshold"],
            "best_threshold_accuracy": best["accuracy"],
            "best_threshold_sensitivity": best["sensitivity"],
            "best_threshold_specificity": best["specificity"],
            "best_threshold_balanced_accuracy": best["balanced_accuracy"],
            "prediction_rows": prediction_rows,
        }
    )
    return metrics


def _write_metrics_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "train_binary_loss",
        "train_ordinal_loss",
        "train_auroc",
        "train_pr_auc",
        "val_loss",
        "val_binary_loss",
        "val_ordinal_loss",
        "val_auroc",
        "val_pr_auc",
        "val_ordinal_mae",
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
        "selection_metric",
        "selection_value",
        "best_threshold",
        "best_threshold_balanced_accuracy",
        "best_epoch_auroc",
        "best_epoch_pr_auc",
        "best_epoch_ordinal_mae",
        "fixed_threshold_balanced_accuracy",
        "best_checkpoint",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _selection_value(metrics: dict[str, object], metric: str) -> float:
    if metric == "loss":
        return -float(metrics["loss"])
    if metric == "best_balanced_accuracy":
        return float(metrics["best_threshold_balanced_accuracy"])
    return float(metrics[metric])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train mask-aware pre-NAC TRG response model.")
    parser.add_argument("--data-root", type=Path, default=Path(r"H:\COCA\CMS_npy_simple"))
    parser.add_argument("--label-xlsx", type=Path, default=Path(r"H:\COCA\label.xlsx"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "nac_trg_response")
    parser.add_argument("--binary-rule", choices=["trg0_vs_123", "trg01_vs_23"], default="trg01_vs_23")
    parser.add_argument("--allow-missing-mask", action="store_true")
    parser.add_argument("--cv-folds", type=int, default=1)
    parser.add_argument("--target-shape", type=_parse_shape, default=(64, 128, 128))
    parser.add_argument("--train-crop-shape", type=_parse_shape, default=None)
    parser.add_argument("--tumor-centered-crop-prob", type=float, default=1.0)
    parser.add_argument("--ring-radius", type=int, default=7)
    parser.add_argument("--mask-as-input", action="store_true")
    parser.add_argument("--window", type=float, nargs=2, default=(-150.0, 250.0))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--prepare-cache-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lambda-ordinal", type=float, default=0.3)
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--flip-prob", type=float, default=0.0)
    parser.add_argument("--intensity-jitter", type=float, default=0.0)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--selection-metric", choices=["auroc", "pr_auc", "best_balanced_accuracy", "loss"], default="auroc")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(progress=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.cv_folds < 1:
        raise SystemExit("--cv-folds must be >= 1")
    if args.ring_radius < 0:
        raise SystemExit("--ring-radius must be >= 0")
    if not 0.0 <= args.tumor_centered_crop_prob <= 1.0:
        raise SystemExit("--tumor-centered-crop-prob must be between 0 and 1")
    if not 0.0 <= args.flip_prob <= 1.0:
        raise SystemExit("--flip-prob must be between 0 and 1")
    if args.intensity_jitter < 0:
        raise SystemExit("--intensity-jitter must be >= 0")
    if args.noise_std < 0:
        raise SystemExit("--noise-std must be >= 0")
    if args.lambda_ordinal < 0:
        raise SystemExit("--lambda-ordinal must be >= 0")
    if args.prepare_cache_only and args.cache_dir is None:
        raise SystemExit("--prepare-cache-only requires --cache-dir")


def _prepare_caches(args: argparse.Namespace, rows: Sequence[ManifestRow]) -> None:
    if args.cache_dir is None:
        return
    crop_shapes = [args.train_crop_shape]
    if args.train_crop_shape is not None:
        crop_shapes.append(None)
    for crop_shape in crop_shapes:
        summary = prepare_cache(
            rows,
            cache_root=args.cache_dir,
            target_shape=args.target_shape,
            crop_shape=crop_shape,
            tumor_centered_crop_prob=args.tumor_centered_crop_prob,
            ring_radius=args.ring_radius,
            window=tuple(args.window),
            mask_as_input=args.mask_as_input,
            rebuild=args.rebuild_cache,
            seed=args.seed,
            progress=args.progress,
        )
        print(
            f"Cache ready: {summary['cache_dir']} "
            f"written={summary['written']} skipped={summary['skipped']} total={summary['total']}"
        )


def _make_dataset(args: argparse.Namespace, rows: Sequence[ManifestRow], *, training: bool, seed: int) -> Dataset:
    crop_shape = args.train_crop_shape if training else None
    if args.cache_dir is not None:
        return CachedNACResponseDataset(
            rows,
            cache_root=args.cache_dir,
            target_shape=args.target_shape,
            crop_shape=crop_shape,
            tumor_centered_crop_prob=args.tumor_centered_crop_prob,
            ring_radius=args.ring_radius,
            window=tuple(args.window),
            mask_as_input=args.mask_as_input,
            augment=training and args.augment,
            flip_prob=args.flip_prob,
            intensity_jitter=args.intensity_jitter,
            noise_std=args.noise_std,
            seed=args.seed,
        )
    return NACResponseDataset(
        rows,
        target_shape=args.target_shape,
        crop_shape=crop_shape,
        tumor_centered_crop_prob=args.tumor_centered_crop_prob,
        ring_radius=args.ring_radius,
        window=tuple(args.window),
        mask_as_input=args.mask_as_input,
        augment=training and args.augment,
        flip_prob=args.flip_prob,
        intensity_jitter=args.intensity_jitter,
        noise_std=args.noise_std,
        seed=seed,
    )


def _train_one_split(
    args: argparse.Namespace,
    *,
    train_rows: Sequence[ManifestRow],
    val_rows: Sequence[ManifestRow],
    output_dir: Path,
    fold_index: int | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest_csv(train_rows, output_dir / "train_manifest.csv")
    _write_manifest_csv(val_rows, output_dir / "val_manifest.csv")
    split_seed = args.seed if fold_index is None else args.seed + fold_index
    train_dataset = _make_dataset(args, train_rows, training=True, seed=split_seed)
    val_dataset = _make_dataset(args, val_rows, training=False, seed=split_seed)
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
    model = TRGResponseNet(
        in_channels=3 if args.mask_as_input else 1,
        stats_dim=ROI_STATS_DIM,
        base_channels=args.base_channels,
        depth=args.depth,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    train_pos = sum(row.binary_label == 1 for row in train_rows)
    train_neg = sum(row.binary_label == 0 for row in train_rows)
    pos_weight_value = 1.0 if args.balanced_sampler else train_neg / max(train_pos, 1)
    pos_weight = torch.tensor([pos_weight_value], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    metrics_csv = output_dir / "metrics.csv"
    if metrics_csv.exists():
        metrics_csv.unlink()

    best_value = float("-inf")
    best_metrics: dict[str, object] = {}
    best_epoch = 0
    best_path = output_dir / "best.pt"
    fold_label = f"fold={fold_index + 1} " if fold_index is not None else ""
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            lambda_ordinal=args.lambda_ordinal,
            pos_weight=pos_weight,
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
                lambda_ordinal=args.lambda_ordinal,
                pos_weight=pos_weight,
                progress=args.progress,
                desc=f"{fold_label}epoch {epoch:03d}/{args.epochs} val",
            )
        prediction_rows = val_metrics.pop("prediction_rows")
        threshold = float(val_metrics["best_threshold"])
        _write_prediction_csv(prediction_rows, output_dir / f"val_predictions_epoch_{epoch:03d}.csv", threshold=threshold)
        _write_metrics_row(
            metrics_csv,
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_binary_loss": train_metrics["binary_loss"],
                "train_ordinal_loss": train_metrics["ordinal_loss"],
                "train_auroc": train_metrics["auroc"],
                "train_pr_auc": train_metrics["pr_auc"],
                "val_loss": val_metrics["loss"],
                "val_binary_loss": val_metrics["binary_loss"],
                "val_ordinal_loss": val_metrics["ordinal_loss"],
                "val_auroc": val_metrics["auroc"],
                "val_pr_auc": val_metrics["pr_auc"],
                "val_ordinal_mae": val_metrics["ordinal_mae"],
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
            f"{fold_label}epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_auroc={val_metrics['auroc']:.4f} "
            f"val_pr_auc={val_metrics['pr_auc']:.4f} "
            f"val_best_bal_acc={val_metrics['best_threshold_balanced_accuracy']:.4f} "
            f"val_ord_mae={val_metrics['ordinal_mae']:.4f}"
        )
        current_value = _selection_value(val_metrics, args.selection_metric)
        if current_value > best_value:
            best_value = current_value
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            _write_prediction_csv(prediction_rows, output_dir / "best_val_predictions.csv", threshold=threshold)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "fold": fold_index + 1 if fold_index is not None else None,
                    "selection_metric": args.selection_metric,
                    "selection_value": current_value,
                    "val_metrics": best_metrics,
                    "best_threshold": val_metrics["best_threshold"],
                    "best_val_predictions_csv": str(output_dir / "best_val_predictions.csv"),
                },
                best_path,
            )
    print(f"{fold_label}Best checkpoint: {best_path} {args.selection_metric}={best_value:.4f}")
    return {
        "fold": fold_index + 1 if fold_index is not None else 0,
        "train_rows": len(train_rows),
        "train_positive": train_pos,
        "train_negative": train_neg,
        "val_rows": len(val_rows),
        "val_positive": sum(row.binary_label == 1 for row in val_rows),
        "val_negative": sum(row.binary_label == 0 for row in val_rows),
        "best_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "selection_value": best_value,
        "best_threshold": best_metrics.get("best_threshold", ""),
        "best_threshold_balanced_accuracy": best_metrics.get("best_threshold_balanced_accuracy", ""),
        "best_epoch_auroc": best_metrics.get("auroc", ""),
        "best_epoch_pr_auc": best_metrics.get("pr_auc", ""),
        "best_epoch_ordinal_mae": best_metrics.get("ordinal_mae", ""),
        "fixed_threshold_balanced_accuracy": best_metrics.get("balanced_accuracy", ""),
        "best_checkpoint": str(best_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _validate_args(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rows = build_manifest(
        args.data_root,
        args.label_xlsx,
        binary_rule=args.binary_rule,
        require_mask=not args.allow_missing_mask,
    )
    if args.max_samples:
        rows = rows[: args.max_samples]
    if len(rows) < 4:
        raise SystemExit("Need at least 4 usable labeled rows.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest_csv(rows, args.output_dir / "manifest.csv")
    print(f"Manifest: {manifest_summary(rows)}")

    if args.cache_dir is not None and not args.dry_run:
        _prepare_caches(args, rows)
        if args.prepare_cache_only:
            return 0

    if args.cv_folds > 1:
        positives = sum(row.binary_label == 1 for row in rows)
        negatives = sum(row.binary_label == 0 for row in rows)
        if args.cv_folds > positives or args.cv_folds > negatives:
            raise SystemExit(f"--cv-folds too large for class counts: positive={positives}, negative={negatives}")
        splits = _stratified_kfold_splits(rows, n_splits=args.cv_folds, seed=args.seed)
        for index, (train_rows, val_rows) in enumerate(splits, start=1):
            print(f"Fold {index}/{args.cv_folds}: train={manifest_summary(train_rows)}, val={manifest_summary(val_rows)}")
        if args.dry_run:
            return 0
        cv_rows = []
        for fold_index, (train_rows, val_rows) in enumerate(splits):
            cv_rows.append(
                _train_one_split(
                    args,
                    train_rows=train_rows,
                    val_rows=val_rows,
                    output_dir=args.output_dir / f"fold_{fold_index + 1}",
                    fold_index=fold_index,
                )
            )
        _write_cv_summary(args.output_dir / "cv_summary.csv", cv_rows)
        values = torch.tensor([float(row["selection_value"]) for row in cv_rows], dtype=torch.float32)
        aurocs = torch.tensor([float(row["best_epoch_auroc"]) for row in cv_rows], dtype=torch.float32)
        pr_aucs = torch.tensor([float(row["best_epoch_pr_auc"]) for row in cv_rows], dtype=torch.float32)
        print(
            f"CV summary: folds={len(cv_rows)} "
            f"mean_selection={values.mean().item():.4f} "
            f"std_selection={values.std(unbiased=False).item():.4f} "
            f"mean_auroc={aurocs.mean().item():.4f} "
            f"mean_pr_auc={pr_aucs.mean().item():.4f}"
        )
        return 0

    train_rows, val_rows = _stratified_split(rows, val_fraction=args.val_fraction, seed=args.seed)
    print(f"Split: train={manifest_summary(train_rows)}, val={manifest_summary(val_rows)}")
    if args.dry_run:
        return 0
    _train_one_split(args, train_rows=train_rows, val_rows=val_rows, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
