---
name: tabular-nelder-mead-threshold-optimization
description: Use scipy Nelder-Mead simplex to optimize regression-to-ordinal thresholds maximizing quadratic weighted kappa on OOF predictions
---

# Nelder-Mead Threshold Optimization

## Overview

When a regression model predicts continuous scores that must be discretized into ordinal classes, the rounding boundaries directly impact QWK. Instead of grid search or manual tuning, use Nelder-Mead simplex optimization to find optimal thresholds on OOF predictions. Gradient-free, fast, and handles the non-differentiable QWK objective natively.

## Quick Start

```python
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score
import numpy as np

def threshold_rounder(preds, thresholds):
    return np.where(preds < thresholds[0], 0,
            np.where(preds < thresholds[1], 1,
             np.where(preds < thresholds[2], 2, 3)))

def neg_qwk(thresholds, y_true, oof_preds):
    rounded = threshold_rounder(oof_preds, thresholds)
    return -cohen_kappa_score(y_true, rounded, weights='quadratic')

result = minimize(neg_qwk, x0=[0.5, 1.5, 2.5],
                  args=(y_true, oof_preds), method='Nelder-Mead')
optimal_thresholds = result.x
test_preds = threshold_rounder(test_raw, optimal_thresholds)
```

## Workflow

1. Train a regression model with K-fold CV, collect OOF predictions
2. Define `threshold_rounder` for N-1 boundaries (N classes)
3. Define objective as negative QWK (minimize = maximize QWK)
4. Run `scipy.optimize.minimize` with `method='Nelder-Mead'` and initial guess at class midpoints
5. Apply optimized thresholds to test predictions

## Key Decisions

- **Initial guess**: use class midpoints (0.5, 1.5, 2.5 for 4 classes) — Nelder-Mead is sensitive to starting point
- **Why Nelder-Mead**: QWK is non-differentiable, gradient-based methods fail; simplex handles discrete jumps well
- **OOF only**: always optimize on OOF to avoid overfitting thresholds to training data
- **Multiple restarts**: run 3-5 times with jittered x0 and take best result for robustness

## References

- [CMI | Tuning | Ensemble of solutions](https://www.kaggle.com/code/batprem/cmi-tuning-ensemble-of-solutions)
- [CMI | Best Single Model](https://www.kaggle.com/code/abdmental01/cmi-best-single-model)
