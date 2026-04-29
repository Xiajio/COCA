from coca_trg.train import build_arg_parser
from coca_trg.train import _validate_args


def test_train_cli_enables_progress_by_default_and_can_disable_it():
    parser = build_arg_parser()

    default_args = parser.parse_args([])
    disabled_args = parser.parse_args(["--no-progress"])

    assert default_args.progress is True
    assert disabled_args.progress is False


def test_train_cli_allows_cache_with_segmentation_auxiliary_loss():
    parser = build_arg_parser()
    args = parser.parse_args(["--cache-dir", "cache", "--lambda-seg", "0.2"])

    _validate_args(args)


def test_train_cli_disables_tumor_feature_fusion_only_when_requested():
    parser = build_arg_parser()

    default_args = parser.parse_args([])
    disabled_args = parser.parse_args(["--no-fusion-features"])

    assert default_args.fusion_features is True
    assert disabled_args.fusion_features is False
