---
name: cv-3d-encoder-2d-decoder-segmentation
description: 3D ResNet encoder extracts volumetric features, pools depth dimension, then feeds into a 2D UNet/FPN decoder for segmentation
---

# 3D Encoder + 2D Decoder Segmentation

## Overview

For volumetric data where the target is a 2D segmentation mask (e.g., ink detection on CT slices, organ segmentation from MRI), a hybrid architecture uses a 3D CNN encoder to capture inter-slice context, then pools the depth dimension from each feature map and feeds the resulting 2D feature maps into a standard 2D UNet/FPN decoder. This combines 3D spatial understanding with efficient 2D decoding.

## Quick Start

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSegModel(nn.Module):
    def __init__(self, encoder_3d, encoder_dims, upscale=4):
        super().__init__()
        self.encoder = encoder_3d  # 3D ResNet returning multi-scale features
        self.decoder = FPNDecoder(encoder_dims, upscale)

    def forward(self, x):
        if x.ndim == 4:
            x = x.unsqueeze(1)  # (B, 1, D, H, W)
        feat_maps_3d = self.encoder(x)  # list of (B, C, D', H', W')
        feat_maps_2d = [f.mean(dim=2) for f in feat_maps_3d]  # pool depth
        return self.decoder(feat_maps_2d)

class FPNDecoder(nn.Module):
    def __init__(self, dims, upscale):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dims[i]+dims[i-1], dims[i-1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(dims[i-1]), nn.ReLU(inplace=True))
            for i in range(1, len(dims))])
        self.logit = nn.Conv2d(dims[0], 1, 1)
        self.up = nn.Upsample(scale_factor=upscale, mode='bilinear')

    def forward(self, features):
        for i in range(len(features)-1, 0, -1):
            up = F.interpolate(features[i], scale_factor=2, mode='bilinear')
            features[i-1] = self.convs[i-1](torch.cat([features[i-1], up], 1))
        return self.up(self.logit(features[0]))
```

## Workflow

1. Pass volumetric input through 3D ResNet encoder (multi-scale feature extraction)
2. Mean-pool depth dimension from each scale's feature map: (B,C,D,H,W) → (B,C,H,W)
3. Feed 2D feature maps into FPN/UNet decoder with skip connections
4. Output 2D segmentation mask at original spatial resolution

## Key Decisions

- **Depth pooling**: mean is default; max preserves strongest activations; attention-weighted is best but heavier
- **Encoder**: 3D ResNet-18/34 is efficient; deeper models need more GPU memory
- **Upscale factor**: match encoder's total spatial downsampling (typically 4x or 8x)
- **vs pure 2.5D**: this captures true 3D features; 2.5D (slices as channels) is faster but less expressive

## References

- [3D ResNet baseline [inference]](https://www.kaggle.com/code/yoyobar/3d-resnet-baseline-inference)
