---
name: cv-dual-view-reshape-forward
description: Reshape dual-view stacked channels into doubled batch dimension for shared backbone, then concatenate with tabular features for classification
---

# Dual-View Reshape Forward

## Overview

When two camera views (e.g., Endzone + Sideline) are stacked as channels in a single tensor, a shared backbone can process both views by reshaping: split channels in half, double the batch size, run the backbone once, then reshape back. This avoids duplicating the backbone while preserving view-specific features. Tabular features (tracking data, distances) are processed through a separate MLP and concatenated before the final classifier.

## Quick Start

```python
import torch
import torch.nn as nn
import timm

class DualViewModel(nn.Module):
    def __init__(self, model_name='tf_efficientnetv2_s', n_frames=13, n_features=18):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True,
                                          num_classes=500, in_chans=n_frames)
        self.mlp = nn.Sequential(
            nn.Linear(n_features, 64), nn.LayerNorm(64),
            nn.ReLU(), nn.Dropout(0.2))
        self.fc = nn.Linear(500 * 2 + 64, 1)

    def forward(self, img, feature):
        b, c, h, w = img.shape
        img = img.reshape(b * 2, c // 2, h, w)   # split views
        img = self.backbone(img).reshape(b, -1)    # (B, 500*2)
        feature = self.mlp(feature)                # (B, 64)
        return self.fc(torch.cat([img, feature], dim=1))

pred = model(dual_view_tensor, tracking_features)
```

## Workflow

1. Stack temporal frames from two views as channels: (B, 2*N_frames, H, W)
2. Reshape to (2*B, N_frames, H, W) — each view becomes a separate batch element
3. Forward through shared backbone → (2*B, embed_dim)
4. Reshape back to (B, 2*embed_dim) — concatenated view features
5. Concatenate with MLP-processed tabular features and classify

## Key Decisions

- **Shared vs separate backbones**: shared halves parameters; separate allows view-specific learning
- **Channel split**: assumes equal frames per view; pad if asymmetric
- **Tabular branch**: LayerNorm + Dropout stabilizes the small MLP
- **vs late fusion**: this mid-fusion approach lets the classifier see both view embeddings jointly

## References

- [NFL 2.5D CNN](https://www.kaggle.com/code/royalacecat/nfl-2-5d-cnn)
- [NFL 2.5D CNN Baseline [Inference]](https://www.kaggle.com/code/zzy990106/nfl-2-5d-cnn-baseline-inference)
