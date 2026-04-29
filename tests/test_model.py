import torch

from coca_trg.models import COCAForTRG


def test_coca_for_trg_returns_segmentation_and_binary_classification_outputs():
    model = COCAForTRG(in_channels=1, seg_classes=2, base_channels=4, depth=2)
    image = torch.randn(2, 1, 16, 32, 32)

    outputs = model(image)

    assert outputs["seg_logits"].shape == (2, 2, 16, 32, 32)
    assert outputs["cls_logit"].shape == (2,)
    assert len(outputs["decoder_features"]) == 2


def test_coca_for_trg_can_fuse_tumor_probability_volume_features():
    model = COCAForTRG(in_channels=1, seg_classes=2, base_channels=4, depth=2, fusion_features=True)
    image = torch.randn(2, 1, 16, 32, 32)

    outputs = model(image)

    assert outputs["cls_logit"].shape == (2,)
    assert outputs["fusion_features"].shape == (2, 4)
    assert outputs["tumor_volume_fraction"].shape == (2,)
    assert torch.all(outputs["fusion_features"] >= 0)
    assert torch.all(outputs["fusion_features"] <= 1)


def test_tumor_probability_features_include_probability_std_not_duplicate_mean():
    tumor_prob = torch.tensor(
        [
            [
                [[0.0, 0.5], [1.0, 1.0]],
            ],
            [
                [[0.2, 0.2], [0.2, 0.2]],
            ],
        ],
        dtype=torch.float32,
    )

    features = COCAForTRG._tumor_probability_features(tumor_prob)
    flat = tumor_prob.flatten(start_dim=1)
    expected = torch.stack(
        [
            flat.mean(dim=1),
            flat.max(dim=1).values,
            flat.std(dim=1, unbiased=False),
            (flat > 0.5).float().mean(dim=1),
        ],
        dim=1,
    )

    assert torch.allclose(features, expected)
