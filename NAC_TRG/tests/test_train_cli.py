import pytest

from nac_trg.train import build_arg_parser, _validate_args


def test_train_cli_exposes_response_model_options():
    parser = build_arg_parser()

    args = parser.parse_args(
        [
            "--cv-folds",
            "5",
            "--ring-radius",
            "7",
            "--lambda-ordinal",
            "0.3",
            "--mask-as-input",
            "--balanced-sampler",
            "--augment",
            "--cache-dir",
            r"H:\COCA\NAC_TRG\cache",
            "--rebuild-cache",
            "--prepare-cache-only",
            "--selection-metric",
            "auroc",
        ]
    )

    assert args.cv_folds == 5
    assert args.ring_radius == 7
    assert args.lambda_ordinal == 0.3
    assert args.mask_as_input is True
    assert args.balanced_sampler is True
    assert args.augment is True
    assert str(args.cache_dir) == r"H:\COCA\NAC_TRG\cache"
    assert args.rebuild_cache is True
    assert args.prepare_cache_only is True
    assert args.selection_metric == "auroc"
    _validate_args(args)


def test_prepare_cache_only_requires_cache_dir():
    parser = build_arg_parser()
    args = parser.parse_args(["--prepare-cache-only"])

    with pytest.raises(SystemExit):
        _validate_args(args)
