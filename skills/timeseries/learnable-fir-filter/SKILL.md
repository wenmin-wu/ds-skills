---
name: timeseries-learnable-fir-filter
description: Initialize a depthwise Conv1d with FIR filter coefficients as a trainable high-pass/low-pass filter for sensor signal preprocessing
domain: timeseries
---

# Learnable FIR Filter

## Overview

Instead of fixed signal preprocessing, initialize a depthwise 1D convolution with FIR filter coefficients (e.g. high-pass from scipy.signal.firwin), then let it fine-tune during training. The model learns the optimal frequency response for the task while starting from a sensible prior.

## Quick Start

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import firwin

class LearnableFIR(nn.Module):
    def __init__(self, n_channels, numtaps=33, cutoff=1.0, fs=200.0):
        super().__init__()
        # Initialize with FIR high-pass coefficients
        fir_coeff = firwin(numtaps, cutoff=cutoff, fs=fs, pass_zero=False)
        kernel = torch.tensor(fir_coeff, dtype=torch.float32)
        kernel = kernel.view(1, 1, -1).repeat(n_channels, 1, 1)
        self.weight = nn.Parameter(kernel)
        self.n_channels = n_channels
        self.pad = numtaps // 2

    def forward(self, x):  # x: (B, C, T)
        filtered = F.conv1d(x, self.weight, padding=self.pad,
                           groups=self.n_channels)
        residual = x - filtered  # complementary filter (low-pass)
        return torch.cat([filtered, residual], dim=1)  # both HPF + LPF
```

## Key Decisions

- **Depthwise conv**: one filter per channel, not cross-channel — preserves sensor independence
- **FIR initialization**: starts from known good filter, fine-tunes to task — much better than random init
- **HPF + LPF output**: concatenate filtered and residual signals as complementary features
- **numtaps=33**: odd number, ~165ms window at 200Hz — captures relevant motion frequencies

## References

- Source: [cmi-detect-behavior-with-sensor-data](https://www.kaggle.com/code/nina2025/cmi-detect-behavior-with-sensor-data)
- Competition: CMI - Detect Behavior with Sensor Data
