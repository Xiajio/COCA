from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .metadata import ManifestRow


ROI_STATS_DIM = 14


def _to_depth_first(array: np.ndarray) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {array.shape}")
    if array.shape[0] == array.shape[1] and array.shape[2] != array.shape[0]:
        return np.moveaxis(array, -1, 0)
    return array


def _window_to_unit_range(volume: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    lo, hi = window
    if hi <= lo:
        raise ValueError(f"Invalid CT window {window}; upper bound must exceed lower bound")
    clipped = np.clip(volume.astype(np.float32, copy=False), lo, hi)
    return ((clipped - lo) / (hi - lo) * 2.0 - 1.0).astype(np.float32, copy=False)


def _resize_image(volume: np.ndarray, target_shape: tuple[int, int, int]) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(volume).copy()).float()[None, None]
    return F.interpolate(tensor, size=target_shape, mode="trilinear", align_corners=False)[0, 0]


def _resize_mask(mask: np.ndarray, target_shape: tuple[int, int, int]) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(mask).copy()).float()[None, None]
    return F.interpolate(tensor, size=target_shape, mode="nearest")[0, 0].long()


def _center_crop_or_pad(
    array: np.ndarray,
    *,
    center: tuple[int, int, int],
    crop_shape: tuple[int, int, int],
    pad_value: float | int = 0,
) -> np.ndarray:
    result = np.full(crop_shape, pad_value, dtype=array.dtype)
    source_slices = []
    target_slices = []
    for size, center_index, crop_size in zip(array.shape, center, crop_shape):
        start = int(center_index) - crop_size // 2
        end = start + crop_size
        source_start = max(start, 0)
        source_end = min(end, size)
        target_start = source_start - start
        target_end = target_start + (source_end - source_start)
        source_slices.append(slice(source_start, source_end))
        target_slices.append(slice(target_start, target_end))
    result[tuple(target_slices)] = array[tuple(source_slices)]
    return result


def _random_center(shape: tuple[int, int, int], rng: np.random.Generator) -> tuple[int, int, int]:
    return tuple(int(rng.integers(0, dim)) for dim in shape)


def _random_foreground_center(mask: np.ndarray, rng: np.random.Generator) -> tuple[int, int, int] | None:
    foreground = np.argwhere(mask > 0)
    if foreground.size == 0:
        return None
    index = int(rng.integers(0, len(foreground)))
    return tuple(int(value) for value in foreground[index])


def make_peritumor_ring(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return torch.zeros_like(mask, dtype=torch.long)
    mask_float = mask.float()[None, None]
    dilated = F.max_pool3d(mask_float, kernel_size=2 * radius + 1, stride=1, padding=radius)[0, 0]
    return ((dilated > 0) & (mask == 0)).long()


def _empty_stats() -> list[float]:
    return [0.0, 0.0, 0.0, 0.0]


def _masked_stats(values: torch.Tensor, mask: torch.Tensor, *, include_minmax: bool) -> list[float]:
    selected = values[mask.bool()]
    if selected.numel() == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] if include_minmax else _empty_stats()
    mean = float(selected.mean())
    std = float(selected.std(unbiased=False))
    p10 = float(torch.quantile(selected, 0.10))
    p90 = float(torch.quantile(selected, 0.90))
    if include_minmax:
        return [mean, std, float(selected.min()), float(selected.max()), p10, p90]
    return [mean, std, p10, p90]


def roi_statistics(image: torch.Tensor, tumor_mask: torch.Tensor, peritumor_ring: torch.Tensor) -> torch.Tensor:
    tumor = tumor_mask.bool()
    volume_fraction = float(tumor.float().mean())
    bbox = [0.0, 0.0, 0.0]
    coords = torch.nonzero(tumor, as_tuple=False)
    if coords.numel() > 0:
        spans = coords.max(dim=0).values - coords.min(dim=0).values + 1
        shape = torch.tensor(tumor_mask.shape, dtype=torch.float32)
        bbox = (spans.float() / shape).tolist()
    stats = [
        volume_fraction,
        float(bbox[0]),
        float(bbox[1]),
        float(bbox[2]),
        *_masked_stats(image, tumor_mask, include_minmax=True),
        *_masked_stats(image, peritumor_ring, include_minmax=False),
    ]
    return torch.tensor(stats, dtype=torch.float32)


def _augment(
    image: torch.Tensor,
    tumor_mask: torch.Tensor,
    peritumor_ring: torch.Tensor,
    *,
    rng: np.random.Generator,
    flip_prob: float,
    intensity_jitter: float,
    noise_std: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    for image_dim, mask_dim in ((2, 1), (3, 2)):
        if flip_prob > 0 and rng.random() < flip_prob:
            image = torch.flip(image, dims=(image_dim,))
            tumor_mask = torch.flip(tumor_mask, dims=(mask_dim,))
            peritumor_ring = torch.flip(peritumor_ring, dims=(mask_dim,))
    if intensity_jitter > 0:
        scale = float(rng.uniform(1.0 - intensity_jitter, 1.0 + intensity_jitter))
        shift = float(rng.uniform(-intensity_jitter, intensity_jitter))
        image = image.mul(scale).add(shift)
    if noise_std > 0:
        noise = rng.normal(0.0, noise_std, size=tuple(image.shape)).astype(np.float32)
        image = image + torch.from_numpy(noise)
    return image.clamp(-1.0, 1.0), tumor_mask, peritumor_ring


class NACResponseDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[ManifestRow],
        *,
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
        if not 0.0 <= tumor_centered_crop_prob <= 1.0:
            raise ValueError("tumor_centered_crop_prob must be between 0 and 1")
        if ring_radius < 0:
            raise ValueError("ring_radius must be >= 0")
        if not 0.0 <= flip_prob <= 1.0:
            raise ValueError("flip_prob must be between 0 and 1")
        self.rows = list(rows)
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
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        volume = _to_depth_first(np.asarray(np.load(row.image_path, mmap_mode="r")))
        mask = np.zeros_like(volume, dtype=np.uint8)
        if row.mask_path is not None and row.mask_shape_matches:
            mask = _to_depth_first(np.asarray(np.load(row.mask_path, mmap_mode="r"))).astype(np.uint8, copy=False)

        if self.crop_shape is not None:
            center = None
            if self.rng.random() < self.tumor_centered_crop_prob:
                center = _random_foreground_center(mask, self.rng)
            if center is None:
                center = _random_center(tuple(int(dim) for dim in volume.shape), self.rng)
            volume = _center_crop_or_pad(volume, center=center, crop_shape=self.crop_shape)
            mask = _center_crop_or_pad(mask, center=center, crop_shape=self.crop_shape)

        image = _resize_image(_window_to_unit_range(volume, self.window), self.target_shape).unsqueeze(0)
        tumor_mask = _resize_mask(mask, self.target_shape).clamp(0, 1)
        peritumor_ring = make_peritumor_ring(tumor_mask, self.ring_radius)
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
        stats = roi_statistics(image[0], tumor_mask, peritumor_ring)
        model_image = image
        if self.mask_as_input:
            model_image = torch.cat([image, tumor_mask.float().unsqueeze(0), peritumor_ring.float().unsqueeze(0)], dim=0)
        return {
            "image": model_image,
            "tumor_mask": tumor_mask,
            "peritumor_ring": peritumor_ring,
            "roi_stats": stats,
            "label": torch.tensor(float(row.binary_label), dtype=torch.float32),
            "trg": torch.tensor(row.trg, dtype=torch.long),
            "case_id": row.case_id,
        }
