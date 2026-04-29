from pathlib import Path
import shutil

import numpy as np
import torch

from coca_trg.cache import CachedTRGVolumeDataset, cache_path_for, mask_cache_path_for, prepare_cache
from coca_trg.metadata import ManifestRow


def _test_dir(name: str) -> Path:
    path = Path("tmp_test_dir") / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def test_prepare_cache_writes_resized_float16_volume_and_dataset_loads_it():
    tmp_path = _test_dir("cache")
    image_path = tmp_path / "image.npy"
    np.save(image_path, np.linspace(-1000, 1000, 6 * 12 * 10).reshape(6, 12, 10))
    row = ManifestRow(
        case_id="case-a",
        image_path=image_path,
        mask_path=None,
        trg=1,
        binary_label=1,
        image_shape=(6, 12, 10),
        mask_shape=None,
        mask_shape_matches=False,
    )
    cache_dir = tmp_path / "cache"

    written = prepare_cache(
        [row],
        cache_dir=cache_dir,
        target_shape=(8, 16, 12),
        window=(-500, 500),
        dtype="float16",
    )

    cache_path = cache_path_for(
        row,
        cache_dir=cache_dir,
        target_shape=(8, 16, 12),
        window=(-500, 500),
        dtype="float16",
    )
    cached = np.load(cache_path)
    dataset = CachedTRGVolumeDataset([row], cache_dir=cache_dir, target_shape=(8, 16, 12), window=(-500, 500))
    sample = dataset[0]

    assert written == 1
    assert cached.shape == (8, 16, 12)
    assert cached.dtype == np.float16
    assert sample["image"].shape == (1, 8, 16, 12)
    assert sample["label"].item() == 1.0


def test_prepare_cache_writes_mask_and_cached_dataset_loads_auxiliary_mask():
    tmp_path = _test_dir("cache_mask")
    image_path = tmp_path / "image.npy"
    mask_path = tmp_path / "mask.npy"
    np.save(image_path, np.linspace(-1000, 1000, 6 * 12 * 10).reshape(6, 12, 10))
    mask = np.zeros((6, 12, 10), dtype=np.uint8)
    mask[2:4, 4:8, 3:6] = 1
    np.save(mask_path, mask)
    row = ManifestRow(
        case_id="case-mask",
        image_path=image_path,
        mask_path=mask_path,
        trg=2,
        binary_label=0,
        image_shape=(6, 12, 10),
        mask_shape=(6, 12, 10),
        mask_shape_matches=True,
    )
    cache_dir = tmp_path / "cache"

    prepare_cache(
        [row],
        cache_dir=cache_dir,
        target_shape=(8, 16, 12),
        window=(-500, 500),
        include_masks=True,
        progress=False,
    )

    cached_mask = np.load(
        mask_cache_path_for(
            row,
            cache_dir=cache_dir,
            target_shape=(8, 16, 12),
            window=(-500, 500),
        )
    )
    dataset = CachedTRGVolumeDataset(
        [row],
        cache_dir=cache_dir,
        target_shape=(8, 16, 12),
        window=(-500, 500),
        include_masks=True,
    )
    sample = dataset[0]

    assert cached_mask.shape == (8, 16, 12)
    assert cached_mask.dtype == np.uint8
    assert sample["mask"].shape == (8, 16, 12)
    assert sample["mask"].dtype == torch.long
    assert sample["has_mask"].item() is True
