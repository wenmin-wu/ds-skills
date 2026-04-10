---
name: timeseries-mean-residual-decomposition
description: Decompose multi-output prediction into a global mean (1D model) plus per-channel residuals (2D model) with quadrature uncertainty
domain: timeseries
---

# Mean-Residual Decomposition

## Overview

For multi-output regression where outputs share a common baseline (e.g. spectrum = mean depth + per-wavelength variation), train two models: one predicts the global mean from a summary signal, another predicts per-channel residuals. Final prediction = mean + residuals. Uncertainty propagates in quadrature.

## Quick Start

```python
import numpy as np

def predict_decomposed(mean_model, residual_model, summary_input, detail_input,
                        n_mc=100):
    """Predict via mean + residual decomposition with MC uncertainty."""
    # Mean prediction (1D)
    mean_preds = np.stack([mean_model(summary_input, training=True)
                           for _ in range(n_mc)])
    mean_val = mean_preds.mean(axis=0)
    mean_std = mean_preds.std(axis=0)
    # Residual prediction (2D)
    res_preds = np.stack([residual_model(detail_input, training=True)
                          for _ in range(n_mc)])
    res_val = res_preds.mean(axis=0)
    res_std = res_preds.std(axis=0)
    # Combine
    prediction = res_val + mean_val[:, np.newaxis]
    uncertainty = np.sqrt(mean_std[:, np.newaxis]**2 + res_std**2)
    return prediction, uncertainty
```

## Key Decisions

- **Why decompose**: mean model sees high-SNR summary; residual model focuses on subtle variations
- **Quadrature propagation**: `σ_total = √(σ_mean² + σ_residual²)` — independent uncertainties
- **Summary signal**: e.g. white light curve (sum over channels), spatial mean, broadband average
- **Residuals are zero-centered**: subtract mean from targets before training residual model

## References

- Source: [host-starter-solution](https://www.kaggle.com/code/gordonyip/host-starter-solution)
- Competition: NeurIPS - Ariel Data Challenge 2024
