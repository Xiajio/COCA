from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from coca_trg.metadata import build_manifest


def _write_volume(path: Path, shape=(4, 8, 6), dtype=np.float32):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.zeros(shape, dtype=dtype))


def _write_labels(path: Path):
    df = pd.DataFrame(
        {
            "CRF号": [1, 2, 3, 4],
            "TRG（4分类）": [0, 1, 3, np.nan],
        }
    )
    df.to_excel(path, index=False)


def _test_dir(name: str) -> Path:
    path = Path("tmp_test_dir") / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def test_build_manifest_defaults_to_trg01_positive_and_skips_missing_trg():
    tmp_path = _test_dir("metadata_trg01_default")
    data_root = tmp_path / "CMS_npy_simple"
    _write_volume(data_root / "Dicom" / "1.npy")
    _write_volume(data_root / "Dicom" / "2.npy")
    _write_volume(data_root / "Dicom" / "3.npy")
    _write_volume(data_root / "Dicom" / "4.npy")
    _write_volume(data_root / "labels" / "1_label.npy", dtype=np.uint8)
    _write_volume(data_root / "labels" / "2-tumor-label_label.npy", dtype=np.uint8)
    _write_volume(data_root / "labels" / "3_label.npy", shape=(4, 8, 5), dtype=np.uint8)
    _write_volume(data_root / "labels" / "4_label.npy", dtype=np.uint8)
    label_xlsx = tmp_path / "label.xlsx"
    _write_labels(label_xlsx)

    manifest = build_manifest(data_root=data_root, label_xlsx=label_xlsx)

    assert [row.case_id for row in manifest] == ["1", "2", "3"]
    assert [row.binary_label for row in manifest] == [1, 1, 0]
    assert manifest[0].mask_shape_matches is True
    assert manifest[1].mask_path.name == "2-tumor-label_label.npy"
    assert manifest[2].mask_shape_matches is False


def test_build_manifest_can_use_trg0_pcr_rule():
    tmp_path = _test_dir("metadata_trg0")
    data_root = tmp_path / "CMS_npy_simple"
    for case_id in ["1", "2", "3"]:
        _write_volume(data_root / "Dicom" / f"{case_id}.npy")
    label_xlsx = tmp_path / "label.xlsx"
    _write_labels(label_xlsx)

    manifest = build_manifest(
        data_root=data_root,
        label_xlsx=label_xlsx,
        binary_rule="trg0_vs_123",
    )

    assert [row.case_id for row in manifest] == ["1", "2", "3"]
    assert [row.binary_label for row in manifest] == [1, 0, 0]
