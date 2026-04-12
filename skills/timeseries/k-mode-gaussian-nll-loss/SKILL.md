---
name: timeseries-k-mode-gaussian-nll-loss
description: Negative log-likelihood loss over K isotropic-Gaussian trajectory modes with per-mode confidences and logsumexp stability
---

## Overview

Single-trajectory regression (MSE) assumes one future is correct. Real driving, motion, and forecasting problems are *multimodal* — a vehicle might turn left or go straight, and both are plausible. The standard fix is to predict K trajectories plus K confidence weights, and optimize the negative log-likelihood of the ground truth under a mixture of K isotropic Gaussians (variance = 1). The math simplifies to `-logsumexp(log(conf) - 0.5 * Σ error²)` along the K axis, which rewards any mode that nails the ground truth while letting the confidences learn the mixture weights. Used in Lyft Motion Prediction, Waymo Open, Argoverse, and most AV trajectory benchmarks.

## Quick Start

```python
import torch
import numpy as np

def neg_multi_log_likelihood(gt, pred, confidences, avails):
    """
    gt:          (B, T, 2)    ground-truth trajectory
    pred:        (B, K, T, 2) K candidate trajectories
    confidences: (B, K)       softmax over modes
    avails:      (B, T)       1/0 mask for valid future steps
    """
    gt = gt.unsqueeze(1)                    # (B, 1, T, 2)
    avails = avails[:, None, :, None]       # (B, 1, T, 1)
    err = torch.sum(((gt - pred) * avails) ** 2, dim=(2, 3))  # (B, K)

    with np.errstate(divide='ignore'):
        err = torch.log(confidences) - 0.5 * err               # (B, K)

    # logsumexp trick for numerical stability
    max_val, _ = err.max(dim=1, keepdim=True)
    err = -(torch.log(torch.sum(torch.exp(err - max_val), dim=1, keepdim=True))
            + max_val).squeeze(1)
    return err.mean()
```

## Workflow

1. Model outputs `(pred, confidences)` where `pred` has shape `(B, K, T, 2)` and `confidences` is softmax-normalized over K
2. Multiply per-timestep squared error by the availability mask to ignore missing steps
3. Compute per-mode log-likelihood `log(conf) - 0.5 * sum(err²)`
4. Apply the log-sum-exp trick: subtract the max before exp, add it back after the log
5. Negate and average over the batch

## Key Decisions

- **Isotropic unit variance**: simpler than learning per-mode covariance, and almost always enough. Covariance regression is brittle without a lot of data.
- **Softmax confidences, not sigmoid**: the mixture weights must sum to 1.
- **Logsumexp is mandatory**: without it, raw exp underflows to 0 and gradients vanish on well-fit batches.
- **vs. winner-take-all MSE**: WTA only updates the best mode, freezing the others. NLL trains all modes jointly and produces calibrated confidences.

## References

- [Lyft: Complete train and prediction pipeline](https://www.kaggle.com/code/pestipeti/lyft-complete-train-and-prediction-pipeline)
- [Pytorch Baseline - Train](https://www.kaggle.com/code/pestipeti/pytorch-baseline-train)
