from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Encoder3D(nn.Module):
    def __init__(self, *, in_channels: int = 1, base_channels: int = 16, depth: int = 3) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        blocks = []
        ch_in = in_channels
        for i in range(depth):
            ch_out = base_channels * (2**i)
            blocks.append(ConvBlock3D(ch_in, ch_out))
            ch_in = ch_out
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = ch_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index < len(self.blocks) - 1:
                x = F.max_pool3d(x, kernel_size=2, stride=2)
        return x


class MaskedAveragePooling3D(nn.Module):
    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim == 4:
            mask = mask.unsqueeze(1)
        mask = mask.float()
        if mask.shape[-3:] != features.shape[-3:]:
            mask = F.interpolate(mask, size=features.shape[-3:], mode="nearest")
        weighted = features * mask
        denom = mask.sum(dim=(2, 3, 4)).clamp_min(1.0)
        return weighted.sum(dim=(2, 3, 4)) / denom


class TRGResponseNet(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 1,
        stats_dim: int = 14,
        base_channels: int = 16,
        depth: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = Encoder3D(in_channels=in_channels, base_channels=base_channels, depth=depth)
        self.mask_pool = MaskedAveragePooling3D()
        self.stats_mlp = nn.Sequential(
            nn.Linear(stats_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        fused_dim = self.encoder.out_channels * 3 + hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.binary_head = nn.Linear(hidden_dim, 1)
        self.ordinal_head = nn.Linear(hidden_dim, 3)

    def forward(
        self,
        *,
        image: torch.Tensor,
        tumor_mask: torch.Tensor,
        peritumor_ring: torch.Tensor,
        roi_stats: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features = self.encoder(image)
        global_feature = features.mean(dim=(2, 3, 4))
        tumor_feature = self.mask_pool(features, tumor_mask)
        peritumor_feature = self.mask_pool(features, peritumor_ring)
        stats_feature = self.stats_mlp(roi_stats.float())
        fused = torch.cat([global_feature, tumor_feature, peritumor_feature, stats_feature], dim=1)
        embedding = self.fusion(fused)
        binary_logit = self.binary_head(embedding).squeeze(1)
        ordinal_logits = self.ordinal_head(embedding)
        return {
            "binary_logit": binary_logit,
            "binary_prob": torch.sigmoid(binary_logit),
            "ordinal_logits": ordinal_logits,
            "ordinal_prob": torch.sigmoid(ordinal_logits),
            "embedding": embedding,
        }
