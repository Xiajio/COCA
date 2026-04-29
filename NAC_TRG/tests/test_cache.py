from pathlib import Path
import shutil

import numpy as np

from nac_trg.cache import CachedNACResponseDataset, prepare_cache
from nac_trg.metadata import ManifestRow


def _test_dir(name: str) -> Path:
    path = Path("tmp_test_dir") / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def _row(image_path: Path, mask_path: Path) -> ManifestRow:
    return ManifestRow(
        case_id="case-cache",
        image_path=image_path,
        mask_path=mask_path,
        trg=1,
        binary_label=1,
        image_shape=(8, 10, 12),
        mask_shape=(8, 10, 12),
        mask_shape_matches=True,
    )


def test_prepare_cache_writes_preprocessed_case_and_cached_dataset_reads_it():
    tmp_path = _test_dir("response_cache")
    image = np.zeros((8, 10, 12), dtype=np.float32)
    image[3:5, 4:7, 5:8] = 120.0
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[3:5, 4:7, 5:8] = 1
    image_path = tmp_path / "image.npy"
    mask_path = tmp_path / "mask_label.npy"
    np.save(image_path, image)
    np.save(mask_path, mask)

    row = _row(image_path, mask_path)
    cache_root = tmp_path / "cache"
    summary = prepare_cache(
        [row],
        cache_root=cache_root,
        target_shape=(6, 8, 10),
        crop_shape=None,
        tumor_centered_crop_prob=1.0,
        ring_radius=1,
        window=(-100.0, 200.0),
        mask_as_input=False,
        rebuild=False,
        seed=123,
        progress=False,
    )

    assert summary["total"] == 1
    assert summary["written"] == 1

    dataset = CachedNACResponseDataset(
        [row],
        cache_root=cache_root,
        target_shape=(6, 8, 10),
        crop_shape=None,
        tumor_centered_crop_prob=1.0,
        ring_radius=1,
        window=(-100.0, 200.0),
        mask_as_input=False,
        seed=123,
    )
    sample = dataset[0]

    assert sample["image"].shape == (1, 6, 8, 10)
    assert sample["tumor_mask"].shape == (6, 8, 10)
    assert sample["peritumor_ring"].shape == (6, 8, 10)
    assert sample["roi_stats"].shape == (14,)
    assert sample["label"].item() == 1.0
    assert sample["trg"].item() == 1
    assert sample["case_id"] == "case-cache"


def test_cached_dataset_can_return_mask_and_ring_input_channels():
    tmp_path = _test_dir("response_cache_channels")
    image = np.ones((8, 10, 12), dtype=np.float32)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[2:6, 3:8, 4:9] = 1
    image_path = tmp_path / "image.npy"
    mask_path = tmp_path / "mask_label.npy"
    np.save(image_path, image)
    np.save(mask_path, mask)

    row = _row(image_path, mask_path)
    cache_root = tmp_path / "cache"
    prepare_cache(
        [row],
        cache_root=cache_root,
        target_shape=(8, 10, 12),
        crop_shape=None,
        tumor_centered_crop_prob=1.0,
        ring_radius=2,
        window=(-150.0, 250.0),
        mask_as_input=True,
        rebuild=False,
        seed=123,
        progress=False,
    )

    dataset = CachedNACResponseDataset(
        [row],
        cache_root=cache_root,
        target_shape=(8, 10, 12),
        crop_shape=None,
        tumor_centered_crop_prob=1.0,
        ring_radius=2,
        window=(-150.0, 250.0),
        mask_as_input=True,
        seed=123,
    )
    sample = dataset[0]

    assert sample["image"].shape == (3, 8, 10, 12)
