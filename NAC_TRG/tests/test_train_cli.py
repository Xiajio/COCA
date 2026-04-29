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
    assert args.selection_metric == "auroc"
    _validate_args(args)
