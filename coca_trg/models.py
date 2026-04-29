from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _center_crop_or_pad_to_match(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    target = reference.shape[-3:]
    result = x
    pads: list[int] = []

    for current, wanted in zip(reversed(result.shape[-3:]), reversed(target)):
        diff = wanted - current
        before = max(diff // 2, 0)
        after = max(diff - before, 0)
        pads.extend([before, after])
    if any(pads):
        result = F.pad(result, pads)

    slices = [slice(None), slice(None)]
    for current, wanted in zip(result.shape[-3:], target):
        if current == wanted:
            slices.append(slice(None))
        else:
            start = max((current - wanted) // 2, 0)
            slices.append(slice(start, start + wanted))
    return result[tuple(slices)]


class UNet3D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 2,
        base_channels: int = 8,
        depth: int = 3,
        return_decoder_features: bool = False,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.return_decoder_features = return_decoder_features

        encoders = []
        pools = []
        ch_in = in_channels
        for i in range(depth):
            ch_out = base_channels * (2**i)
            encoders.append(ConvBlock3D(ch_in, ch_out))
            pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            ch_in = ch_out
        self.enc_blocks = nn.ModuleList(encoders)
        self.pools = nn.ModuleList(pools)

        self.bottleneck = ConvBlock3D(base_channels * (2 ** (depth - 1)), base_channels * (2**depth))

        up_layers = []
        decoders = []
        for i in reversed(range(depth)):
            ch_skip = base_channels * (2**i)
            ch_up = base_channels * (2 ** (i + 1))
            up_layers.append(nn.ConvTranspose3d(ch_up, ch_skip, kernel_size=2, stride=2))
            decoders.append(ConvBlock3D(ch_skip + ch_skip, ch_skip))
        self.up_layers = nn.ModuleList(up_layers)
        self.dec_blocks = nn.ModuleList(decoders)
        self.seg_head = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        skips = []
        for encoder, pool in zip(self.enc_blocks, self.pools):
            x = encoder(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        decoder_features = []
        for up_layer, decoder, skip in zip(self.up_layers, self.dec_blocks, reversed(skips)):
            x = up_layer(x)
            x = _center_crop_or_pad_to_match(x, skip)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
            decoder_features.append(x)

        seg_logits = self.seg_head(x)
        if self.return_decoder_features:
            return seg_logits, decoder_features
        return seg_logits


class COCAForTRG(nn.Module):
    """COCA Stage-II style model adapted for TRG binary classification."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        seg_classes: int = 2,
        base_channels: int = 8,
        depth: int = 3,
        classifier_hidden: int = 64,
        dropout: float = 0.2,
        fusion_features: bool = True,
    ) -> None:
        super().__init__()
        self.fusion_features = fusion_features
        self.diag_unet = UNet3D(
            in_channels=in_channels,
            out_channels=seg_classes,
            base_channels=base_channels,
            depth=depth,
            return_decoder_features=True,
        )
        classifier_in = sum(base_channels * (2**i) for i in range(depth))
        if fusion_features:
            self.fusion_mlp = nn.Sequential(
                nn.Linear(4, 16),
                nn.ReLU(inplace=True),
                nn.Linear(16, 16),
                nn.ReLU(inplace=True),
            )
            classifier_in += 16
        else:
            self.fusion_mlp = None
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1),
        )

    @staticmethod
    def _tumor_probability_features(tumor_prob: torch.Tensor) -> torch.Tensor:
        flat = tumor_prob.flatten(start_dim=1)
        mean_prob = flat.mean(dim=1)
        max_prob = flat.max(dim=1).values
        std_prob = flat.std(dim=1, unbiased=False)
        hard_volume_fraction = (flat > 0.5).float().mean(dim=1)
        return torch.stack(
            [
                mean_prob,
                max_prob,
                std_prob,
                hard_volume_fraction,
            ],
            dim=1,
        ).clamp(0.0, 1.0)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        seg_logits, decoder_features = self.diag_unet(image)
        pooled = [feature.mean(dim=(2, 3, 4)) for feature in decoder_features]
        embedding = torch.cat(pooled, dim=1)
        seg_prob = torch.softmax(seg_logits, dim=1)
        tumor_prob = seg_prob[:, 1]
        fusion_features = self._tumor_probability_features(tumor_prob)
        if self.fusion_mlp is not None:
            embedding = torch.cat([embedding, self.fusion_mlp(fusion_features)], dim=1)
        cls_logit = self.classifier(embedding).squeeze(1)
        tumor_volume_fraction = fusion_features[:, 3]
        tumor_volume = (tumor_prob > 0.5).sum(dim=(1, 2, 3))
        return {
            "seg_logits": seg_logits,
            "seg_prob": seg_prob,
            "tumor_prob": tumor_prob,
            "decoder_features": decoder_features,
            "fusion_features": fusion_features,
            "cls_logit": cls_logit,
            "cls_prob": torch.sigmoid(cls_logit),
            "tumor_volume_fraction": tumor_volume_fraction,
            "tumor_volume": tumor_volume,
        }
