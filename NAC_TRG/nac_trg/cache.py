from __future__ import annotations

from collections.abc import Sequence
import hashlib
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .dataset import NACResponseDataset, _augment
from .metadata import ManifestRow


def _number_token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _shape_token(shape: tuple[int, int, int] | None) -> str:
    if shape is None:
        return "none"
    return "x".join(str(int(part)) for part in shape)


def _safe_case_id(case_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(case_id))


def _case_seed(base_seed: int, case_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{case_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def cache_directory(
    cache_root: str | Path,
    *,
    target_shape: tuple[int, int, int],
    crop_shape: tuple[int, int, int] | None,
    tumor_centered_crop_prob: float,
    ring_radius: int,
    window: tuple[float, float],
    seed: int,
) -> Path:
    lo, hi = window
    name = (
        f"target_{_shape_token(target_shape)}"
        f"__crop_{_shape_token(crop_shape)}"
        f"__tumorprob_{_number_token(tumor_centered_crop_prob)}"
        f"__ring_{int(ring_radius)}"
        f"__win_{_number_token(lo)}_{_number_token(hi)}"
        f"__seed_{int(seed)}"
    )
    return Path(cache_root) / name


def cache_file(
    cache_root: str | Path,
    row: ManifestRow,
    *,
    target_shape: tuple[int, int, int],
    crop_shape: tuple[int, int, int] | None,
    tumor_centered_crop_prob: float,
    ring_radius: int,
    window: tuple[float, float],
    seed: int,
) -> Path:
    return (
        cache_directory(
            cache_root,
            target_shape=target_shape,
            crop_shape=crop_shape,
            tumor_centered_crop_prob=tumor_centered_crop_prob,
            ring_radius=ring_radius,
            window=window,
            seed=seed,
        )
        / f"{_safe_case_id(row.case_id)}.pt"
    )


def _torch_load(path: Path) -> dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def prepare_cache(
    rows: Sequence[ManifestRow],
    *,
    cache_root: str | Path,
    target_shape: tuple[int, int, int],
    crop_shape: tuple[int, int, int] | None,
    tumor_centered_crop_prob: float,
    ring_radius: int,
    window: tuple[float, float],
    mask_as_input: bool = False,
    rebuild: bool = False,
    seed: int = 2026,
    progress: bool = True,
) -> dict[str, object]:
    del mask_as_input
    cache_dir = cache_directory(
        cache_root,
        target_shape=target_shape,
        crop_shape=crop_shape,
        tumor_centered_crop_prob=tumor_centered_crop_prob,
        ring_radius=ring_radius,
        window=window,
        seed=seed,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    iterator = tqdm(rows, desc="prepare-cache", unit="case", leave=False, disable=not progress)
    for row in iterator:
        path = cache_file(
            cache_root,
            row,
            target_shape=target_shape,
            crop_shape=crop_shape,
            tumor_centered_crop_prob=tumor_centered_crop_prob,
            ring_radius=ring_radius,
            window=window,
            seed=seed,
        )
        if path.exists() and not rebuild:
            skipped += 1
            continue
        dataset = NACResponseDataset(
            [row],
            target_shape=target_shape,
            crop_shape=crop_shape,
            tumor_centered_crop_prob=tumor_centered_crop_prob,
            ring_radius=ring_radius,
            window=window,
            mask_as_input=False,
            augment=False,
            seed=_case_seed(seed, row.case_id),
        )
        sample = dataset[0]
        payload = {
            "case_id": row.case_id,
            "image": sample["image"].contiguous(),
            "tumor_mask": sample["tumor_mask"].contiguous(),
            "peritumor_ring": sample["peritumor_ring"].contiguous(),
            "roi_stats": sample["roi_stats"].contiguous(),
            "target_shape": tuple(int(part) for part in target_shape),
            "crop_shape": tuple(int(part) for part in crop_shape) if crop_shape is not None else None,
            "tumor_centered_crop_prob": float(tumor_centered_crop_prob),
            "ring_radius": int(ring_radius),
            "window": tuple(float(value) for value in window),
            "seed": int(seed),
        }
        tmp_path = path.with_suffix(".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
        written += 1
    return {"cache_dir": str(cache_dir), "total": len(rows), "written": written, "skipped": skipped}


class CachedNACResponseDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[ManifestRow],
        *,
        cache_root: str | Path,
        target_shape: tuple[int, int, int] = (64, 128, 128),
        crop_shape: tuple[int, int, int] | None = None,
        tumor_centered_crop_prob: float = 1.0,
        ring_radius: int = 7,
        window: tuple[float, float] = (-150.0, 250.0),
        mask_as_input: bool = False,
        augment: bool = False,
        flip_prob: float = 0.0,
        intensity_jitter: float = 0.0,
        noise_std: float = 0.0,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= flip_prob <= 1.0:
            raise ValueError("flip_prob must be between 0 and 1")
        self.rows = list(rows)
        self.cache_root = Path(cache_root)
        self.target_shape = target_shape
        self.crop_shape = crop_shape
        self.tumor_centered_crop_prob = tumor_centered_crop_prob
        self.ring_radius = ring_radius
        self.window = window
        self.mask_as_input = mask_as_input
        self.augment = augment
        self.flip_prob = flip_prob
        self.intensity_jitter = intensity_jitter
        self.noise_std = noise_std
        self.seed = 2026 if seed is None else seed
        self.rng = np.random.default_rng(self.seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        path = cache_file(
            self.cache_root,
            row,
            target_shape=self.target_shape,
            crop_shape=self.crop_shape,
            tumor_centered_crop_prob=self.tumor_centered_crop_prob,
            ring_radius=self.ring_radius,
            window=self.window,
            seed=self.seed,
        )
        if not path.exists():
            raise FileNotFoundError(f"Missing preprocessed cache for case {row.case_id}: {path}")
        payload = _torch_load(path)
        image = payload["image"].clone().float()
        tumor_mask = payload["tumor_mask"].clone().long()
        peritumor_ring = payload["peritumor_ring"].clone().long()
        roi_stats = payload["roi_stats"].clone().float()
        if self.augment:
            image, tumor_mask, peritumor_ring = _augment(
                image,
                tumor_mask,
                peritumor_ring,
                rng=self.rng,
                flip_prob=self.flip_prob,
                intensity_jitter=self.intensity_jitter,
                noise_std=self.noise_std,
            )
        model_image = image
        if self.mask_as_input:
            model_image = torch.cat([image, tumor_mask.float().unsqueeze(0), peritumor_ring.float().unsqueeze(0)], dim=0)
        return {
            "image": model_image,
            "tumor_mask": tumor_mask,
            "peritumor_ring": peritumor_ring,
            "roi_stats": roi_stats,
            "label": torch.tensor(float(row.binary_label), dtype=torch.float32),
            "trg": torch.tensor(row.trg, dtype=torch.long),
            "case_id": row.case_id,
        }
