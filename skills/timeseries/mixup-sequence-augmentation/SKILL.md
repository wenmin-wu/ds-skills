---
name: timeseries-mixup-sequence-augmentation
description: Apply MixUp augmentation to padded time-series batches with Beta-distributed lambda and soft label mixing
domain: timeseries
---

# MixUp Sequence Augmentation

## Overview

MixUp linearly interpolates between pairs of training samples and their labels using a Beta-distributed lambda. For time series classification, apply it to padded/fixed-length sequences. Reduces overfitting and improves calibration, especially with small datasets.

## Quick Start

```python
import numpy as np
import torch
import torch.nn.functional as F

def mixup_batch(x, y, alpha=0.2):
    """MixUp augmentation for time-series batches.
    
    Args:
        x: (B, T, C) input sequences
        y: (B, num_classes) one-hot labels
        alpha: Beta distribution parameter
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    perm = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[perm]
    y_mix = lam * y + (1 - lam) * y[perm]
    return x_mix, y_mix

def soft_cross_entropy(pred, soft_targets):
    """Cross-entropy loss for soft (mixed) labels."""
    return -torch.sum(soft_targets * F.log_softmax(pred, dim=1), dim=1).mean()
```

## Key Decisions

- **alpha=0.2**: mild mixing — lambda typically near 0 or 1, preserving sample identity
- **Soft CE loss required**: standard CE expects hard labels; use manual log_softmax + weighted sum
- **Shuffle per-epoch**: re-permute mixing pairs each epoch for diversity

## References

- Source: [cmi25-imu-thm-tof-tf-blendingmodel-lb-82](https://www.kaggle.com/code/hideyukizushi/cmi25-imu-thm-tof-tf-blendingmodel-lb-82)
- Competition: CMI - Detect Behavior with Sensor Data
