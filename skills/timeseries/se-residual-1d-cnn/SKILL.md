---
name: timeseries-se-residual-1d-cnn
description: 1D ResNet block with Squeeze-and-Excitation channel attention for temporal sensor feature extraction
domain: timeseries
---

# SE-Residual 1D CNN

## Overview

Squeeze-and-Excitation (SE) blocks learn per-channel importance weights via global average pooling → FC → sigmoid gating. Combined with residual connections in 1D CNNs, they selectively amplify informative sensor channels while maintaining gradient flow. Effective for IMU, EEG, and other multi-channel time series.

## Quick Start

```python
import torch
import torch.nn as nn

class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: (B, C, T)
        scale = x.mean(dim=2)  # (B, C)
        scale = self.fc(scale).unsqueeze(2)  # (B, C, 1)
        return x * scale

class ResidualSEBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SEBlock1D(out_ch)
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        return self.relu(out + self.shortcut(x))
```

## Key Decisions

- **reduction=4**: balance between expressiveness and parameter cost
- **Global avg pool**: captures channel-wise statistics across full temporal extent
- **Residual connection**: ensures gradient flow even when SE gate is near-zero

## References

- Source: [just-changed-the-ensemble-weights](https://www.kaggle.com/code/sasaleaf/just-changed-the-ensemble-weights)
- Competition: CMI - Detect Behavior with Sensor Data
