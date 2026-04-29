import torch

from nac_trg.losses import ordinal_targets_from_trg, response_loss


def test_ordinal_targets_encode_ordered_trg_thresholds():
    trg = torch.tensor([0, 1, 2, 3])

    targets = ordinal_targets_from_trg(trg)

    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )
    assert torch.equal(targets, expected)


def test_response_loss_combines_binary_and_ordinal_losses():
    outputs = {
        "binary_logit": torch.tensor([0.0, 1.0]),
        "ordinal_logits": torch.zeros(2, 3),
    }
    labels = torch.tensor([0.0, 1.0])
    trgs = torch.tensor([3, 1])

    losses = response_loss(outputs, labels, trgs, lambda_ordinal=0.3)

    assert losses["loss"].ndim == 0
    assert losses["binary_loss"].ndim == 0
    assert losses["ordinal_loss"].ndim == 0
    assert torch.allclose(losses["loss"], losses["binary_loss"] + 0.3 * losses["ordinal_loss"])
