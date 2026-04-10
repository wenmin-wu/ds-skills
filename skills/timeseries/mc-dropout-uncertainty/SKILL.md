---
name: timeseries-mc-dropout-uncertainty
description: Estimate prediction uncertainty via Monte Carlo Dropout — run inference N times with dropout active and compute mean/std
domain: timeseries
---

# MC Dropout Uncertainty

## Overview

Monte Carlo Dropout keeps dropout active at inference time and runs N forward passes. The mean gives a smoothed prediction; the standard deviation gives a calibrated uncertainty estimate. No architectural changes needed — just force `training=True` on dropout layers.

## Quick Start

```python
import numpy as np

def mc_dropout_predict(model, X, n_samples=100):
    """Run N stochastic forward passes and return mean + std."""
    preds = np.stack([model(X, training=True).numpy() for _ in range(n_samples)])
    return preds.mean(axis=0), preds.std(axis=0)

# PyTorch version
def mc_predict_torch(model, X, n_samples=100):
    model.train()  # keeps dropout active
    with torch.no_grad():
        preds = torch.stack([model(X) for _ in range(n_samples)])
    return preds.mean(0), preds.std(0)
```

## Key Decisions

- **n_samples=100**: good balance of quality vs compute; 25+ gives usable uncertainty
- **training=True**: the only change needed — forces dropout to sample
- **Quadrature propagation**: combine MC std with other uncertainty sources via `sqrt(σ1² + σ2²)`
- **Works with any dropout model**: no special architecture required

## References

- Source: [host-starter-solution](https://www.kaggle.com/code/gordonyip/host-starter-solution)
- Competition: NeurIPS - Ariel Data Challenge 2024
