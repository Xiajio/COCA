import pytest

from coca_trg.train import _validate_args, build_arg_parser


def test_train_parser_accepts_original_space_tumor_centered_crop_args():
    parser = build_arg_parser()

    args = parser.parse_args(
        [
            "--train-crop-shape",
            "32,96,96",
            "--tumor-centered-crop-prob",
            "0.75",
        ]
    )

    assert args.train_crop_shape == (32, 96, 96)
    assert args.tumor_centered_crop_prob == 0.75


def test_train_rejects_crop_sampling_with_resized_cache():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--cache-dir",
            "cache",
            "--train-crop-shape",
            "32,96,96",
        ]
    )

    with pytest.raises(SystemExit, match="--train-crop-shape"):
        _validate_args(args)


def test_train_parser_accepts_cv_balanced_sampler_and_augmentation_args():
    parser = build_arg_parser()

    args = parser.parse_args(
        [
            "--cv-folds",
            "5",
            "--balanced-sampler",
            "--augment",
            "--flip-prob",
            "0.25",
            "--intensity-jitter",
            "0.1",
            "--noise-std",
            "0.02",
        ]
    )

    assert args.cv_folds == 5
    assert args.balanced_sampler is True
    assert args.augment is True
    assert args.flip_prob == 0.25
    assert args.intensity_jitter == 0.1
    assert args.noise_std == 0.02


def test_train_rejects_invalid_cv_and_augmentation_args():
    parser = build_arg_parser()
    args = parser.parse_args(["--cv-folds", "0"])

    with pytest.raises(SystemExit, match="--cv-folds"):
        _validate_args(args)

    args = parser.parse_args(["--flip-prob", "1.5"])

    with pytest.raises(SystemExit, match="--flip-prob"):
        _validate_args(args)
