---
name: timeseries-gaussian-log-likelihood-metric
description: Evaluate probabilistic forecasts using normalized Gaussian log-likelihood relative to naive and oracle baselines, scoring both mean accuracy and uncertainty calibration
domain: timeseries
---

# Gaussian Log-Likelihood Metric

## Overview

For regression tasks requiring uncertainty estimates, score predictions using Gaussian log-likelihood normalized between a naive baseline and an oracle. The metric rewards both accurate mean predictions AND well-calibrated uncertainty (sigma). Overconfident predictions (small sigma, wrong mean) are penalized heavily. Score ranges from 0 (= naive) to 1 (= oracle).

## Quick Start

```python
import numpy as np
from scipy.stats import norm

def gaussian_log_likelihood_score(y_true, y_pred, sigma_pred,
                                   naive_mean, naive_sigma, sigma_true):
    """Normalized Gaussian log-likelihood metric.
    
    Args:
        y_true: ground truth values (n_samples, n_targets)
        y_pred: predicted means (n_samples, n_targets)
        sigma_pred: predicted uncertainties (n_samples, n_targets)
        naive_mean: baseline mean prediction (n_targets,)
        naive_sigma: baseline uncertainty (n_targets,)
        sigma_true: oracle uncertainty (n_samples, n_targets)
    Returns:
        score in [0, 1] where 1 is perfect
    """
    sigma_pred = np.clip(sigma_pred, 1e-15, None)
    
    gll_pred = np.sum(norm.logpdf(y_true, loc=y_pred, scale=sigma_pred))
    gll_oracle = np.sum(norm.logpdf(y_true, loc=y_true, scale=sigma_true))
    gll_naive = np.sum(norm.logpdf(y_true, loc=naive_mean, scale=naive_sigma))
    
    score = (gll_pred - gll_naive) / (gll_oracle - gll_naive)
    return float(np.clip(score, 0.0, 1.0))
```

## Key Decisions

- **Normalized**: 0 = naive baseline, 1 = oracle; interpretable across datasets
- **Clip sigma**: floor at 1e-15 to prevent -inf from log(0)
- **Joint scoring**: penalizes overconfidence — can't game it with small sigma
- **Naive baseline**: typically training set mean and std per target

## References

- Source: [neurips-adc-25-intro-training](https://www.kaggle.com/code/ahsuna123/neurips-adc-25-intro-training)
- Competition: NeurIPS - Ariel Data Challenge 2025
