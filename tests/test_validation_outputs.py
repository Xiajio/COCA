from pathlib import Path
import shutil

import torch

from coca_trg.train import _make_prediction_rows, _ranking_metrics, _search_best_threshold, _write_cv_summary, _write_metrics_row


def _test_dir(name: str) -> Path:
    path = Path("tmp_test_dir") / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def test_make_prediction_rows_keeps_case_level_probability_and_labels():
    logits = torch.logit(torch.tensor([0.2, 0.8]))
    labels = torch.tensor([0.0, 1.0])
    trgs = torch.tensor([3, 1])
    rows = _make_prediction_rows(
        logits=logits,
        labels=labels,
        trgs=trgs,
        case_ids=["case-neg", "case-pos"],
    )

    assert rows == [
        {"case_id": "case-neg", "probability": 0.2, "label": 0, "trg": 3},
        {"case_id": "case-pos", "probability": 0.8, "label": 1, "trg": 1},
    ]


def test_search_best_threshold_can_improve_over_fixed_point_five():
    probs = torch.tensor([0.2, 0.4, 0.6, 0.7])
    labels = torch.tensor([0.0, 1.0, 1.0, 0.0])

    result = _search_best_threshold(probs, labels)

    assert result["threshold"] == 0.4
    assert result["balanced_accuracy"] == 0.75
    assert result["sensitivity"] == 1.0
    assert result["specificity"] == 0.5


def test_ranking_metrics_report_auroc_and_pr_auc_without_threshold_search():
    probs = torch.tensor([0.1, 0.4, 0.35, 0.8])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])

    metrics = _ranking_metrics(probs, labels)

    assert metrics["auroc"] == 0.75
    assert round(metrics["pr_auc"], 6) == 0.833333


def test_metric_writers_include_ranking_metric_columns():
    tmp_path = _test_dir("metric_writers")
    metrics_path = tmp_path / "metrics.csv"
    _write_metrics_row(
        metrics_path,
        {
            "epoch": 1,
            "train_auroc": 0.61,
            "train_pr_auc": 0.35,
            "val_auroc": 0.71,
            "val_pr_auc": 0.42,
        },
    )

    header = metrics_path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "train_auroc" in header
    assert "train_pr_auc" in header
    assert "val_auroc" in header
    assert "val_pr_auc" in header

    summary_path = tmp_path / "cv_summary.csv"
    _write_cv_summary(
        summary_path,
        [
            {
                "fold": 1,
                "best_epoch_auroc": 0.71,
                "best_epoch_pr_auc": 0.42,
            }
        ],
    )
    summary_header = summary_path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "best_epoch_auroc" in summary_header
    assert "best_epoch_pr_auc" in summary_header
    summary_row = summary_path.read_text(encoding="utf-8").splitlines()[1].split(",")
    assert summary_row[summary_header.index("best_epoch_auroc")] == "0.71"
    assert summary_row[summary_header.index("best_epoch_pr_auc")] == "0.42"
