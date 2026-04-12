---
name: cv-concat-tile-feature-pooling
description: >
  Passes N tiles independently through a shared CNN backbone, concatenates their feature maps spatially, then pools for classification — a lightweight multi-instance learning approach.
---
# Concat Tile Feature Pooling

## Overview

For WSI classification, processing each tile through a shared CNN backbone produces per-tile feature maps. Rather than averaging tile predictions (losing spatial info) or building a full attention-based MIL model, this approach concatenates all tile feature maps along the spatial dimension, then applies adaptive pooling and a classification head. This preserves inter-tile patterns while remaining end-to-end trainable with standard backbones.

## Quick Start

```python
import torch
import torch.nn as nn

class ConcatTileModel(nn.Module):
    def __init__(self, backbone, n_classes=6):
        super().__init__()
        self.enc = nn.Sequential(*list(backbone.children())[:-2])
        nc = list(backbone.children())[-1].in_features
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(nc, 512), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(512, n_classes))

    def forward(self, tiles):
        """tiles: list of N tensors, each (B, 3, H, W)"""
        B = tiles[0].shape[0]
        N = len(tiles)
        x = torch.stack(tiles, dim=1)  # (B, N, 3, H, W)
        x = x.view(B * N, *tiles[0].shape[1:])
        x = self.enc(x)  # (B*N, C, h, w)
        C, h, w = x.shape[1:]
        # Concatenate tile features spatially
        x = x.view(B, N, C, h, w).permute(0, 2, 1, 3, 4)
        x = x.contiguous().view(B, C, N * h, w)
        x = self.head(x)
        return x
```

## Workflow

1. Extract N tiles per slide (e.g., 12-16 tiles of 128x128)
2. Pass all tiles through shared backbone in one batched forward pass
3. Reshape and concatenate feature maps along spatial dimension
4. Apply adaptive pooling + classification head
5. Train end-to-end with standard cross-entropy or ordinal loss

## Key Decisions

- **N tiles**: 12-16 typical; more tiles need more GPU memory
- **vs tile averaging**: Concat preserves spatial arrangement; averaging loses it
- **vs attention MIL**: Simpler, faster, but assumes fixed tile count
- **Backbone**: EfficientNet-B0 or ResNet-34 work well; larger backbones may OOM

## References

- [PANDA concat tile pooling starter](https://www.kaggle.com/code/iafoss/panda-concat-tile-pooling-starter-0-79-lb)
