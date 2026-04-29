from pathlib import Path
import shutil

import numpy as np
import torch

from coca_trg.dataset import TRGVolumeDataset
from coca_trg.metadata import ManifestRow


def _test_dir(name: str) -> Path:
    path = Path("tmp_test_dir") / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def test_dataset_loads_npy_volume_as_channel_first_resized_tensor():
    tmp_path = _test_dir("dataset_image")
    volume = np.linspace(-1000, 1000, num=6 * 12 * 10, dtype=np.float64).reshape(6, 12, 10)
    image_path = tmp_path / "image.npy"
    np.save(image_path, volume)

    row = ManifestRow(
        case_id="case-a",
        image_path=image_path,
        mask_path=None,
        trg=0,
        binary_label=1,
        image_shape=(6, 12, 10),
        mask_shape=None,
        mask_shape_matches=False,
    )

    dataset = TRGVolumeDataset([row], target_shape=(8, 16, 12), window=(-500, 500))
    sample = dataset[0]

    assert sample["image"].shape == (1, 8, 16, 12)
    assert sample["image"].dtype == torch.float32
    assert sample["label"].shape == ()
    assert sample["label"].item() == 1.0
    assert sample["case_id"] == "case-a"
    assert float(sample["image"].min()) >= -1.0
    assert float(sample["image"].max()) <= 1.0


def test_dataset_loads_matching_mask_when_requested():
    tmp_path = _test_dir("dataset_mask")
    image_path = tmp_path / "image.npy"
    mask_path = tmp_path / "mask.npy"
    np.save(image_path, np.ones((4, 8, 6), dtype=np.float32))
    np.save(mask_path, np.ones((4, 8, 6), dtype=np.uint8))
    row = ManifestRow(
        case_id="case-b",
        image_path=image_path,
        mask_path=mask_path,
        trg=2,
        binary_label=0,
        image_shape=(4, 8, 6),
        mask_shape=(4, 8, 6),
        mask_shape_matches=True,
    )

    dataset = TRGVolumeDataset([row], target_shape=(4, 8, 6), include_masks=True)
    sample = dataset[0]

    assert sample["mask"].shape == (4, 8, 6)
    assert sample["has_mask"].item() is True
    assert sample["mask"].dtype == torch.long


def test_dataset_can_sample_positive_patch_centered_on_tumor_mask():
    tmp_path = _test_dir("dataset_tumor_crop")
    image_path = tmp_path / "image.npy"
    mask_path = tmp_path / "mask.npy"
    volume = np.zeros((8, 16, 16), dtype=np.float32)
    mask = np.zeros((8, 16, 16), dtype=np.uint8)
    volume[6, 14, 14] = 200
    mask[6, 14, 14] = 1
    np.save(image_path, volume)
    np.save(mask_path, mask)
    row = ManifestRow(
        case_id="case-crop",
        image_path=image_path,
        mask_path=mask_path,
        trg=0,
        binary_label=1,
        image_shape=(8, 16, 16),
        mask_shape=(8, 16, 16),
        mask_shape_matches=True,
    )

    dataset = TRGVolumeDataset(
        [row],
        target_shape=(4, 8, 8),
        include_masks=True,
        crop_shape=(4, 8, 8),
        tumor_centered_crop_prob=1.0,
        seed=2026,
    )
    sample = dataset[0]

    assert sample["image"].shape == (1, 4, 8, 8)
    assert sample["mask"].shape == (4, 8, 8)
    assert sample["mask"].sum().item() >= 1


def test_dataset_training_augmentation_flips_image_and_mask_together():
    tmp_path = _test_dir("dataset_augment")
    image_path = tmp_path / "image.npy"
    mask_path = tmp_path / "mask.npy"
    volume = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    mask = np.zeros((4, 5, 6), dtype=np.uint8)
    mask[:, 0, 0] = 1
    np.save(image_path, volume)
    np.save(mask_path, mask)
    row = ManifestRow(
        case_id="case-augment",
        image_path=image_path,
        mask_path=mask_path,
        trg=0,
        binary_label=1,
        image_shape=(4, 5, 6),
        mask_shape=(4, 5, 6),
        mask_shape_matches=True,
    )
    plain = TRGVolumeDataset([row], target_shape=(4, 5, 6), window=(0, 119), include_masks=True)[0]
    augmented = TRGVolumeDataset(
        [row],
        target_shape=(4, 5, 6),
        window=(0, 119),
        include_masks=True,
        augment=True,
        flip_prob=1.0,
        intensity_jitter=0.0,
        noise_std=0.0,
        seed=2026,
    )[0]

    assert torch.allclose(augmented["image"], torch.flip(plain["image"], dims=(2, 3)))
    assert torch.equal(augmented["mask"], torch.flip(plain["mask"], dims=(1, 2)))
