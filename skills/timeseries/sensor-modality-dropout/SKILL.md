---
name: timeseries-sensor-modality-dropout
description: Randomly zero out entire sensor modalities during training with a learned gate to handle missing modalities at inference
domain: timeseries
---

# Sensor Modality Dropout

## Overview

In multi-sensor systems, some modalities may be missing at inference time (sensor failure, power saving). Train robustness by randomly zeroing entire modality channels with probability p, while a gating network learns to predict whether a modality is active. At inference, the gate automatically downweights missing modalities.

## Quick Start

```python
import torch
import torch.nn as nn
import numpy as np

class ModalityDropout(nn.Module):
    def __init__(self, n_channels, drop_prob=0.3):
        super().__init__()
        self.drop_prob = drop_prob
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(n_channels, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, x):  # x: (B, C, T)
        gate_val = self.gate(x)  # (B, 1)
        if self.training:
            mask = (torch.rand(x.size(0), 1, 1, device=x.device) > self.drop_prob).float()
            x = x * mask
        return x * gate_val.unsqueeze(2)
```

## Key Decisions

- **drop_prob=0.3**: aggressive enough to learn robustness, not so much that primary signal is lost
- **Learned gate**: sigmoid output scales the modality contribution based on signal quality
- **Drop entire modality**: zero all channels of a sensor, not individual channels — simulates real failure modes
- **Gate supervision optional**: can add auxiliary BCE loss on gate output vs mask target

## References

- Source: [cmi-detect-behavior-with-sensor-data](https://www.kaggle.com/code/nina2025/cmi-detect-behavior-with-sensor-data)
- Competition: CMI - Detect Behavior with Sensor Data
