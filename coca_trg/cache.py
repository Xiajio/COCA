from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .dataset import _augment_image_and_mask, _resize_image, _resize_mask, _to_depth_first, _window_to_unit_range
from .metadata import ManifestRow


def _shape_token(target_shape: tuple[int, int, int]) -> str:
    return "x".join(str(dim) for dim in target_shape)


def _window_token(window: tuple[float, float]) -> str:
    return f"{window[0]:g}_{window[1]:g}".replace("-", "m").replace(".", "p")


def cache_subdir(
    cache_dir: str | Path,
    *,
    target_shape: tuple[int, int, int],
    window: tuple[float, float],
    dtype: str,
) -> Path:
    return Path(cache_dir) / f"DHW_{_shape_token(target_shape)}_win_{_window_token(window)}_{dtype}"


def cache_path_for(
    row: ManifestRow,
    *,
    cache_dir: str | Path,
    target_shape: tuple[int, int, int],
    window: tuple[float, float],
    dtype: str = "float16",
) -> Path:
    return cache_subdir(cache_dir, target_shape=target_shape, window=window, dtype=dtype) / f"{row.case_id}.npy"


def mask_cache_path_for(
    row: ManifestRow,
    *,
    cache_dir: str | Path,
    target_shape: tuple[int, int, int],
    window: tuple[float, float],
) -> Path:
    return cache_subdir(cache_dir, target_shape=target_shape, window=window, dtype="mask_uint8") / f"{row.case_id}.npy"


def _preprocess_image(path: Path, target_shape: tuple[int, int, int], window: tuple[float, float]) -> np.ndarray:
    volume = np.asarray(np.load(path, mmap_mode="r"))
    volume = _to_depth_first(volume)
    volume = _window_to_unit_range(volume, window)
    return _resize_image(volume, target_shape).numpy()


def _preprocess_mask(path: Path, target_shape: tuple[int, int, int]) -> np.ndarray:
    mask = np.asarray(np.load(path, mmap_mode="r"))
    mask = _to_depth_first(mask)
    return _resize_mask(mask, target_shape).numpy().astype(np.uint8, copy=False)


def prepare_cache(
    rows: Sequence[ManifestRow],
    *,
    cache_dir: str | Path,
    target_shape: tuple[int, int, int],
    window: tuple[float, float],
    dtype: str = "float16",
    rebuild: bool = False,
    progress: bool = True,
    include_masks: bool = False,
) -> int:
    if dtype not in {"float16", "float32"}:
        raise ValueError("dtype must be 'float16' or 'float32'")

    out_dir = cache_subdir(cache_dir, target_shape=target_shape, window=window, dtype=dtype)
    mask_out_dir = cache_subdir(cache_dir, target_shape=target_shape, window=window, dtype="mask_uint8")
    out_dir.mkdir(parents=True, exist_ok=True)
    if include_masks:
        mask_out_dir.mkdir(parents=True, exist_ok=True)
    np_dtype = np.float16 if dtype == "float16" else np.float32
    written = 0
    total = len(rows)

    iterator = tqdm(rows, desc="prepare-cache", unit="case", disable=not progress)
    for row in iterator:
        out_path = out_dir / f"{row.case_id}.npy"
        wrote_any = False
        if not out_path.exists() or rebuild:
            image = _preprocess_image(row.image_path, target_shape, window).astype(np_dtype, copy=False)
            np.save(out_path, image)
            written += 1
            wrote_any = True
        if include_masks and row.mask_path is not None and row.mask_shape_matches:
            mask_out_path = mask_out_dir / f"{row.case_id}.npy"
            if not mask_out_path.exists() or rebuild:
                mask = _preprocess_mask(row.mask_path, target_shape)
                np.save(mask_out_path, mask)
                wrote_any = True
        if wrote_any:
            iterator.set_postfix(written=written)

    print(f"Cache ready: {out_dir} written={written} total={total}", flush=True)
    return written


class CachedTRGVolumeDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[ManifestRow],
        *,
        cache_dir: str | Path,
        target_shape: tuple[int, int, int],
        window: tuple[float, float],
        dtype: str = "float16",
        include_masks: bool = False,
        augment: bool = False,
        flip_prob: float = 0.0,
        intensity_jitter: float = 0.0,
        noise_std: float = 0.0,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= flip_prob <= 1.0:
            raise ValueError("flip_prob must be between 0 and 1")
        self.rows = list(rows)
        self.cache_dir = Path(cache_dir)
        self.target_shape = target_shape
        self.window = window
        self.dtype = dtype
        self.include_masks = include_masks
        self.augment = augment
        self.flip_prob = flip_prob
        self.intensity_jitter = intensity_jitter
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        path = cache_path_for(
            row,
            cache_dir=self.cache_dir,
            target_shape=self.target_shape,
            window=self.window,
            dtype=self.dtype,
        )
        image = np.load(path)
        tensor = torch.from_numpy(np.ascontiguousarray(image).copy()).float().unsqueeze(0)
        sample: dict[str, object] = {
            "image": tensor,
            "label": torch.tensor(float(row.binary_label), dtype=torch.float32),
            "trg": torch.tensor(row.trg, dtype=torch.long),
            "case_id": row.case_id,
            "has_mask": torch.tensor(False, dtype=torch.bool),
        }
        if self.include_masks:
            mask_path = mask_cache_path_for(
                row,
                cache_dir=self.cache_dir,
                target_shape=self.target_shape,
                window=self.window,
            )
            if row.mask_shape_matches and mask_path.exists():
                mask = np.load(mask_path)
                sample["mask"] = torch.from_numpy(np.ascontiguousarray(mask).copy()).long()
                sample["has_mask"] = torch.tensor(True, dtype=torch.bool)
            else:
                sample["mask"] = torch.zeros(self.target_shape, dtype=torch.long)
        if self.augment:
            mask = sample.get("mask")
            tensor, augmented_mask = _augment_image_and_mask(
                tensor,
                mask if isinstance(mask, torch.Tensor) else None,
                rng=self.rng,
                flip_prob=self.flip_prob,
                intensity_jitter=self.intensity_jitter,
                noise_std=self.noise_std,
            )
            sample["image"] = tensor
            if augmented_mask is not None:
                sample["mask"] = augmented_mask
        return sample
