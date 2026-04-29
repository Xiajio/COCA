from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ManifestRow:
    case_id: str
    image_path: Path
    mask_path: Path | None
    trg: int
    binary_label: int
    image_shape: tuple[int, int, int]
    mask_shape: tuple[int, int, int] | None
    mask_shape_matches: bool


def _normalize_case_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text or None


def _natural_key(text: str) -> tuple[int, str]:
    return (int(text), text) if text.isdigit() else (10**9, text)


def _find_column(columns: Iterable[object], token: str) -> object:
    for column in columns:
        if token.lower() in str(column).lower():
            return column
    raise ValueError(f"Could not find a column containing {token!r}")


def _case_id_from_mask_name(path: Path) -> str:
    name = path.name.removesuffix("_label.npy")
    name = name.replace("-tumor-label", "").replace("-Tumor-label", "")
    return re.sub(r"_[0-9]+$", "", name)


def _load_shape(path: Path) -> tuple[int, int, int]:
    array = np.load(path, mmap_mode="r")
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D .npy array at {path}, got shape {array.shape}")
    return tuple(int(dim) for dim in array.shape)


def _binary_label(trg: int, binary_rule: str) -> int:
    if binary_rule == "trg0_vs_123":
        if trg == 0:
            return 1
        if trg in (1, 2, 3):
            return 0
    elif binary_rule == "trg01_vs_23":
        if trg in (0, 1):
            return 1
        if trg in (2, 3):
            return 0
    raise ValueError(
        "Unsupported TRG value/rule combination: "
        f"trg={trg!r}, binary_rule={binary_rule!r}"
    )


def discover_images(data_root: Path) -> dict[str, Path]:
    image_dir = data_root / "Dicom"
    return {path.stem: path for path in sorted(image_dir.glob("*.npy"))}


def discover_masks(data_root: Path) -> dict[str, Path]:
    mask_dir = data_root / "labels"
    return {_case_id_from_mask_name(path): path for path in sorted(mask_dir.glob("*_label.npy"))}


def build_manifest(
    data_root: str | Path,
    label_xlsx: str | Path,
    *,
    binary_rule: str = "trg01_vs_23",
) -> list[ManifestRow]:
    """Build image/TRG rows for training.

    The default task treats TRG 0-1 as the positive class and TRG 2-3 as the
    negative class. Rows without image data or without TRG are skipped.
    """

    data_root = Path(data_root)
    label_xlsx = Path(label_xlsx)
    images = discover_images(data_root)
    masks = discover_masks(data_root)

    label_df = pd.read_excel(label_xlsx)
    case_col = _find_column(label_df.columns, "CRF")
    trg_col = _find_column(label_df.columns, "TRG")
    rows: list[ManifestRow] = []

    for _, record in label_df.iterrows():
        case_id = _normalize_case_id(record[case_col])
        if case_id is None or case_id not in images:
            continue

        trg_value = pd.to_numeric(record[trg_col], errors="coerce")
        if pd.isna(trg_value):
            continue
        trg = int(trg_value)
        label = _binary_label(trg, binary_rule)

        image_path = images[case_id]
        mask_path = masks.get(case_id)
        image_shape = _load_shape(image_path)
        mask_shape = _load_shape(mask_path) if mask_path is not None else None
        rows.append(
            ManifestRow(
                case_id=case_id,
                image_path=image_path,
                mask_path=mask_path,
                trg=trg,
                binary_label=label,
                image_shape=image_shape,
                mask_shape=mask_shape,
                mask_shape_matches=mask_shape == image_shape if mask_shape is not None else False,
            )
        )

    return sorted(rows, key=lambda row: _natural_key(row.case_id))


def manifest_summary(rows: list[ManifestRow]) -> dict[str, int]:
    positives = sum(row.binary_label == 1 for row in rows)
    negatives = sum(row.binary_label == 0 for row in rows)
    mask_matches = sum(row.mask_shape_matches for row in rows)
    return {
        "rows": len(rows),
        "positive": positives,
        "negative": negatives,
        "mask_shape_matches": mask_matches,
        "mask_shape_mismatches": len(rows) - mask_matches,
    }
