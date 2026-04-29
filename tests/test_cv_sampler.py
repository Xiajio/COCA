from pathlib import Path

import torch

from coca_trg.metadata import ManifestRow
from coca_trg.train import _balanced_sample_weights, _stratified_kfold_splits


def _row(case_id: str, label: int) -> ManifestRow:
    return ManifestRow(
        case_id=case_id,
        image_path=Path(f"{case_id}.npy"),
        mask_path=None,
        trg=0 if label == 1 else 2,
        binary_label=label,
        image_shape=(4, 4, 4),
        mask_shape=None,
        mask_shape_matches=False,
    )


def test_stratified_kfold_splits_use_each_case_once_for_validation():
    rows = [_row(f"p{i}", 1) for i in range(10)] + [_row(f"n{i}", 0) for i in range(20)]

    folds = _stratified_kfold_splits(rows, n_splits=5, seed=2026)

    assert len(folds) == 5
    validation_ids = []
    for train_rows, val_rows in folds:
        train_ids = {row.case_id for row in train_rows}
        val_ids = {row.case_id for row in val_rows}
        assert train_ids.isdisjoint(val_ids)
        assert sum(row.binary_label == 1 for row in val_rows) == 2
        assert sum(row.binary_label == 0 for row in val_rows) == 4
        validation_ids.extend(val_ids)
    assert sorted(validation_ids) == sorted(row.case_id for row in rows)


def test_balanced_sample_weights_give_each_class_equal_total_weight():
    rows = [_row(f"p{i}", 1) for i in range(2)] + [_row(f"n{i}", 0) for i in range(6)]

    weights = _balanced_sample_weights(rows)
    labels = torch.tensor([row.binary_label for row in rows])

    assert torch.isclose(weights[labels == 1].sum(), weights[labels == 0].sum())
    assert torch.allclose(weights[labels == 1], torch.full((2,), 0.5, dtype=torch.double))
    assert torch.allclose(weights[labels == 0], torch.full((6,), 1.0 / 6.0, dtype=torch.double))
