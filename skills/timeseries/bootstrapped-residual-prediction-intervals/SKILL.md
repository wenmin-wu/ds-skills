---
name: timeseries-bootstrapped-residual-prediction-intervals
description: Generate prediction intervals by repeatedly sampling from model residuals, adding to point forecasts, and taking quantiles across synthetic futures
---

# Bootstrapped Residual Prediction Intervals

## Overview

Given a fitted model's point forecasts and its training residuals, generate prediction intervals without parametric assumptions. Repeatedly sample residuals with replacement, add them to the point forecast to create synthetic future paths, then take quantiles across all paths. This captures the empirical error distribution and naturally adapts interval width to model accuracy.

## Quick Start

```python
import numpy as np

def bootstrap_prediction_intervals(point_forecast, residuals,
                                   quantiles=(0.025, 0.975),
                                   n_bootstrap=1000):
    """Generate prediction intervals via residual bootstrap."""
    n_steps = len(point_forecast)
    paths = np.zeros((n_bootstrap, n_steps))

    for i in range(n_bootstrap):
        sampled = np.random.choice(residuals, size=n_steps, replace=True)
        paths[i] = point_forecast + sampled

    intervals = {}
    for q in quantiles:
        intervals[q] = np.quantile(paths, q, axis=0)
    intervals['median'] = np.median(paths, axis=0)

    return intervals

residuals = y_train - model.predict(X_train)
intervals = bootstrap_prediction_intervals(
    point_forecast=model.predict(X_test),
    residuals=residuals.flatten(),
    quantiles=[0.005, 0.025, 0.165, 0.25, 0.75, 0.835, 0.975, 0.995],
)
```

## Workflow

1. Fit model and compute residuals on training data
2. For each bootstrap iteration, sample N residuals with replacement
3. Add sampled residuals to point forecast to create a synthetic path
4. Repeat 500-1000 times to build a distribution of paths
5. Take quantiles across paths for prediction intervals

## Key Decisions

- **n_bootstrap**: 500-1000 is sufficient; more adds precision at linear cost
- **Residual source**: use out-of-fold residuals to avoid overfitting the intervals
- **Cumulative series**: if forecasting cumulative values, apply cumsum after adding residuals
- **vs parametric**: bootstrap captures skew and fat tails; Gaussian assumes symmetry

## References

- [M5 - Sales Uncertainty Prediction](https://www.kaggle.com/code/allunia/m5-sales-uncertainty-prediction)
