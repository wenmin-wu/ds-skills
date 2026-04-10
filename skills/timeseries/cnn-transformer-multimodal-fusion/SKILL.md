---
name: timeseries-cnn-transformer-multimodal-fusion
description: Process multiple sensor modalities through separate CNN branches then fuse via a transformer with CLS token for classification
domain: timeseries
---

# CNN-Transformer Multimodal Fusion

## Overview

For multi-sensor systems (IMU + thermal + depth), process each modality through its own CNN branch to extract temporal features, concatenate along the feature dimension, prepend a learnable CLS token, and feed into a transformer encoder. The CLS token output is used for classification.

## Quick Start

```python
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

class MultiModalFusion(nn.Module):
    def __init__(self, modality_dims, hidden_dim, n_classes, n_layers=4):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, hidden_dim, 7, padding=3),
                nn.BatchNorm1d(hidden_dim), nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, 5, stride=2, padding=2),
                nn.BatchNorm1d(hidden_dim), nn.GELU(),
            ) for dim in modality_dims
        ])
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim * len(modality_dims)))
        config = BertConfig(hidden_size=hidden_dim * len(modality_dims),
                           num_hidden_layers=n_layers, num_attention_heads=8)
        self.transformer = BertModel(config)
        self.head = nn.Linear(config.hidden_size, n_classes)

    def forward(self, *modalities):
        feats = [branch(m.permute(0, 2, 1)) for branch, m in
                 zip(self.branches, modalities)]
        fused = torch.cat(feats, dim=1).permute(0, 2, 1)  # (B, T, C)
        cls = self.cls_token.expand(fused.size(0), -1, -1)
        x = torch.cat([cls, fused], dim=1)
        out = self.transformer(inputs_embeds=x).last_hidden_state[:, 0]
        return self.head(out)
```

## Key Decisions

- **Separate branches**: each modality has different sampling rates and feature spaces
- **CLS token**: aggregates sequence into a fixed-size vector for classification
- **Stride-2 conv**: reduces temporal resolution before transformer to manage memory

## References

- Source: [cmi-detect-behavior-with-sensor-data](https://www.kaggle.com/code/nina2025/cmi-detect-behavior-with-sensor-data)
- Competition: CMI - Detect Behavior with Sensor Data
