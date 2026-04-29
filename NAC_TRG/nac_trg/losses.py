from __future__ import annotations

import torch
import torch.nn.functional as F


def ordinal_targets_from_trg(trg: torch.Tensor) -> torch.Tensor:
    trg = trg.long()
    thresholds = torch.tensor([0, 1, 2], device=trg.device)
    return (trg[:, None] > thresholds[None, :]).float()


def ordinal_prediction(ordinal_logits: torch.Tensor, *, threshold: float = 0.5) -> torch.Tensor:
    return (torch.sigmoid(ordinal_logits) >= threshold).long().sum(dim=1)


def ordinal_bce_loss(ordinal_logits: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
    targets = ordinal_targets_from_trg(trg).to(dtype=ordinal_logits.dtype)
    return F.binary_cross_entropy_with_logits(ordinal_logits, targets)


def response_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    trgs: torch.Tensor,
    *,
    lambda_ordinal: float = 0.3,
    pos_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    binary_loss = F.binary_cross_entropy_with_logits(
        outputs["binary_logit"],
        labels.float(),
        pos_weight=pos_weight,
    )
    ordinal_loss = ordinal_bce_loss(outputs["ordinal_logits"], trgs)
    total = binary_loss + lambda_ordinal * ordinal_loss
    return {
        "loss": total,
        "binary_loss": binary_loss,
        "ordinal_loss": ordinal_loss,
    }
