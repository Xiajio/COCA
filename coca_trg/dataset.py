from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .metadata import ManifestRow


def _to_depth_first(array: np.ndarray) -> np.ndarray:
    """Convert common saved CT shape [H, W, D] to model shape [D, H, W]."""

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
    resized = F.interpolate(tensor, size=target_shape, mode="trilinear", align_corners=False)
    return resized[0, 0]


def _resize_mask(mask: np.ndarray, target_shape: tuple[int, int, int]) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(mask).copy()).float()[None, None]
    resized = F.interpolate(tensor, size=target_shape, mode="nearest")
    return resized[0, 0].long()


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


def _augment_image_and_mask(
    image: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    rng: np.random.Generator,
    flip_prob: float,
    intensity_jitter: float,
    noise_std: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    for image_dim, mask_dim in ((2, 1), (3, 2)):
        if flip_prob > 0 and rng.random() < flip_prob:
            image = torch.flip(image, dims=(image_dim,))
            if mask is not None:
                mask = torch.flip(mask, dims=(mask_dim,))

    if intensity_jitter > 0:
        scale = float(rng.uniform(1.0 - intensity_jitter, 1.0 + intensity_jitter))
        shift = float(rng.uniform(-intensity_jitter, intensity_jitter))
        image = image.mul(scale).add(shift)

    if noise_std > 0:
        noise = rng.normal(loc=0.0, scale=noise_std, size=tuple(image.shape)).astype(np.float32)
        image = image + torch.from_numpy(noise)

    return image.clamp(-1.0, 1.0), mask


class TRGVolumeDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[ManifestRow],
        *,
        target_shape: tuple[int, int, int] = (64, 128, 128),
        window: tuple[float, float] = (-150.0, 250.0),
        include_masks: bool = False,
        crop_shape: tuple[int, int, int] | None = None,
        tumor_centered_crop_prob: float = 0.0,
        augment: bool = False,
        flip_prob: float = 0.0,
        intensity_jitter: float = 0.0,
        noise_std: float = 0.0,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= tumor_centered_crop_prob <= 1.0:
            raise ValueError("tumor_centered_crop_prob must be between 0 and 1")
        if not 0.0 <= flip_prob <= 1.0:
            raise ValueError("flip_prob must be between 0 and 1")
        self.rows = list(rows)
        self.target_shape = target_shape
        self.window = window
        self.include_masks = include_masks
        self.crop_shape = crop_shape
        self.tumor_centered_crop_prob = tumor_centered_crop_prob
        self.augment = augment
        self.flip_prob = flip_prob
        self.intensity_jitter = intensity_jitter
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        volume = np.asarray(np.load(row.image_path, mmap_mode="r"))
        volume = _to_depth_first(volume)
        mask = None
        load_mask = (self.include_masks or self.crop_shape is not None) and row.mask_path is not None and row.mask_shape_matches
        if load_mask:
            mask = np.asarray(np.load(row.mask_path, mmap_mode="r"))
            mask = _to_depth_first(mask)

        if self.crop_shape is not None:
            center = None
            if (
                row.binary_label == 1
                and mask is not None
                and self.rng.random() < self.tumor_centered_crop_prob
            ):
                center = _random_foreground_center(mask, self.rng)
            if center is None:
                center = _random_center(tuple(int(dim) for dim in volume.shape), self.rng)
            volume = _center_crop_or_pad(volume, center=center, crop_shape=self.crop_shape)
            if mask is not None:
                mask = _center_crop_or_pad(mask, center=center, crop_shape=self.crop_shape)

        volume = _window_to_unit_range(volume, self.window)
        image = _resize_image(volume, self.target_shape).unsqueeze(0)

        sample: dict[str, object] = {
            "image": image,
            "label": torch.tensor(float(row.binary_label), dtype=torch.float32),
            "trg": torch.tensor(row.trg, dtype=torch.long),
            "case_id": row.case_id,
            "has_mask": torch.tensor(False, dtype=torch.bool),
        }

        resized_mask = _resize_mask(mask, self.target_shape) if self.include_masks and mask is not None else None
        if self.augment:
            image, resized_mask = _augment_image_and_mask(
                image,
                resized_mask,
                rng=self.rng,
                flip_prob=self.flip_prob,
                intensity_jitter=self.intensity_jitter,
                noise_std=self.noise_std,
            )
        sample["image"] = image

        if resized_mask is not None:
            sample["mask"] = resized_mask
            sample["has_mask"] = torch.tensor(True, dtype=torch.bool)
        elif self.include_masks:
            sample["mask"] = torch.zeros(self.target_shape, dtype=torch.long)

        return sample
