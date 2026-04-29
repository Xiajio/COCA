from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(seg_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(seg_logits, dim=1)
    target_one_hot = F.one_hot(target.long(), num_classes=seg_logits.shape[1])
    target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()
    dims = (0, 2, 3, 4)
    intersection = (probs * target_one_hot).sum(dim=dims)
    denominator = probs.sum(dim=dims) + target_one_hot.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def segmentation_loss(seg_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(seg_logits, target.long()) + dice_loss(seg_logits, target)
