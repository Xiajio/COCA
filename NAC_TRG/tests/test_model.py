import torch

from nac_trg.models import MaskedAveragePooling3D, TRGResponseNet


def test_masked_average_pooling_uses_only_masked_voxels():
    features = torch.tensor(
        [[[[[1.0, 2.0], [3.0, 4.0]]]]],
        dtype=torch.float32,
    )
    mask = torch.tensor([[[[1, 0], [1, 0]]]], dtype=torch.float32)

    pooled = MaskedAveragePooling3D()(features, mask)

    assert torch.allclose(pooled, torch.tensor([[2.0]]))


def test_trg_response_net_returns_binary_and_ordinal_outputs():
    model = TRGResponseNet(base_channels=4, depth=2, stats_dim=14, hidden_dim=16)
    image = torch.randn(2, 1, 16, 24, 24)
    tumor_mask = torch.zeros(2, 16, 24, 24)
    tumor_mask[:, 4:8, 8:14, 8:14] = 1
    ring = torch.zeros(2, 16, 24, 24)
    ring[:, 3:9, 7:15, 7:15] = 1
    ring = (ring - tumor_mask).clamp_min(0)
    roi_stats = torch.randn(2, 14)

    outputs = model(
        image=image,
        tumor_mask=tumor_mask,
        peritumor_ring=ring,
        roi_stats=roi_stats,
    )

    assert outputs["binary_logit"].shape == (2,)
    assert outputs["binary_prob"].shape == (2,)
    assert outputs["ordinal_logits"].shape == (2, 3)
    assert outputs["ordinal_prob"].shape == (2, 3)
