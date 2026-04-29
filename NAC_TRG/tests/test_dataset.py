from pathlib import Path
import shutil

import numpy as np
import torch

from nac_trg.dataset import NACResponseDataset
from nac_trg.metadata import ManifestRow


def _test_dir(name: str) -> Path:
    path = Path("tmp_test_dir") / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def _row(image_path: Path, mask_path: Path) -> ManifestRow:
    return ManifestRow(
        case_id="case-a",
        image_path=image_path,
        mask_path=mask_path,
        trg=1,
        binary_label=1,
        image_shape=(8, 10, 12),
        mask_shape=(8, 10, 12),
        mask_shape_matches=True,
    )


def test_response_dataset_returns_tumor_ring_and_roi_stats():
    tmp = _test_dir("response_dataset")
    image = np.zeros((8, 10, 12), dtype=np.float32)
    image[3:5, 4:7, 5:8] = 120.0
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[3:5, 4:7, 5:8] = 1
    image_path = tmp / "image.npy"
    mask_path = tmp / "mask_label.npy"
    np.save(image_path, image)
    np.save(mask_path, mask)

    dataset = NACResponseDataset(
        [_row(image_path, mask_path)],
        target_shape=(8, 10, 12),
        ring_radius=1,
        window=(-100.0, 200.0),
    )
    sample = dataset[0]

    assert sample["image"].shape == (1, 8, 10, 12)
    assert sample["tumor_mask"].shape == (8, 10, 12)
    assert sample["peritumor_ring"].shape == (8, 10, 12)
    assert sample["roi_stats"].shape == (14,)
    assert sample["tumor_mask"].sum().item() > 0
    assert sample["peritumor_ring"].sum().item() > 0
    assert torch.logical_and(sample["tumor_mask"].bool(), sample["peritumor_ring"].bool()).sum().item() == 0
    assert sample["label"].item() == 1.0
    assert sample["trg"].item() == 1


def test_response_dataset_can_add_mask_and_ring_as_input_channels():
    tmp = _test_dir("response_dataset_channels")
    image = np.ones((8, 10, 12), dtype=np.float32)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[2:6, 3:8, 4:9] = 1
    image_path = tmp / "image.npy"
    mask_path = tmp / "mask_label.npy"
    np.save(image_path, image)
    np.save(mask_path, mask)

    dataset = NACResponseDataset(
        [_row(image_path, mask_path)],
        target_shape=(8, 10, 12),
        ring_radius=2,
        mask_as_input=True,
    )
    sample = dataset[0]

    assert sample["image"].shape == (3, 8, 10, 12)
