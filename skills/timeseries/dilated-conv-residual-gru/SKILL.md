---
name: timeseries-dilated-conv-residual-gru
description: >
  Combines dilated 1D convolutions for multi-scale receptive fields with residual bidirectional GRU layers for sequence classification.
---
# Dilated Conv + Residual BiGRU

## Overview

Stack dilated 1D convolution blocks (dilation 2^i) to capture local patterns at exponentially increasing receptive fields, then feed into bidirectional GRU layers with residual connections. The CNN extracts local features; the BiGRU captures long-range dependencies. Residual skip connections prevent gradient degradation in deep stacks.

## Quick Start

```python
import torch
import torch.nn as nn

class DilatedConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, n_blocks=4, dropout=0.2):
        super().__init__()
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size,
                          padding=((kernel_size - 1) * 2**i) // 2, dilation=2**i),
                nn.BatchNorm1d(channels),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ) for i in range(n_blocks)
        ])

    def forward(self, x):  # x: (B, C, T)
        return self.blocks(x)

class ResidualBiGRU(nn.Module):
    def __init__(self, hidden_size, n_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x, h=None):  # x: (B, T, H)
        out, h_new = self.gru(x, h)
        out = self.ln(self.fc(out))
        return nn.functional.relu(out) + x, h_new

class SleepModel(nn.Module):
    def __init__(self, in_features, hidden=64, n_classes=2):
        super().__init__()
        self.fc_in = nn.Linear(in_features, hidden)
        self.conv = DilatedConvBlock(hidden)
        self.gru = ResidualBiGRU(hidden)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x, h=None):
        x = self.fc_in(x)                    # (B, T, H)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x, h = self.gru(x, h)
        return self.head(x), h
```

## Workflow

1. Project input features to hidden dimension
2. Pass through dilated conv blocks (captures local multi-scale patterns)
3. Transpose and feed into residual BiGRU (captures long-range dependencies)
4. Linear head outputs per-timestep predictions

## Key Decisions

- **Dilation schedule**: 2^i for i in [0, n_blocks) — receptive field grows exponentially
- **Residual in GRU**: Essential for stacking >2 layers without gradient issues
- **Conv before GRU**: CNN extracts local features; GRU integrates over time
- **LayerNorm > BatchNorm**: In GRU blocks, LayerNorm is more stable for variable-length sequences

## References

- Child Mind Institute - Detect Sleep States (Kaggle)
- Source: [sleep-critical-point-infer](https://www.kaggle.com/code/werus23/sleep-critical-point-infer)
