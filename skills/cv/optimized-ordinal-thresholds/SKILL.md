---
name: cv-optimized-ordinal-thresholds
description: >
  Uses Nelder-Mead optimization to find per-class decision thresholds that maximize Quadratic Weighted Kappa for regression-to-ordinal conversion.
---
# Optimized Ordinal Thresholds

## Overview

Regression models output continuous values that must be mapped to ordinal classes. Simple rounding (0.5, 1.5, 2.5...) is suboptimal because class boundaries aren't equidistant and the evaluation metric (often QWK) isn't linear. Nelder-Mead optimization directly searches for the threshold values that maximize QWK on validation data, typically improving the score by 0.01-0.05 over naive rounding.

## Quick Start

```python
import numpy as np
from functools import partial
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score

class OptimizedRounder:
    def __init__(self, n_classes=5):
        self.coef_ = None
        self.n_classes = n_classes

    def _kappa_loss(self, coef, X, y):
        X_binned = np.digitize(X, coef)
        return -cohen_kappa_score(y, X_binned, weights='quadratic')

    def fit(self, X, y):
        initial = np.arange(0.5, self.n_classes - 0.5)  # [0.5, 1.5, 2.5, 3.5]
        result = minimize(
            partial(self._kappa_loss, X=X, y=y),
            initial, method='nelder-mead'
        )
        self.coef_ = np.sort(result.x)

    def predict(self, X):
        return np.digitize(X, self.coef_)

# Usage
rounder = OptimizedRounder(n_classes=5)
rounder.fit(val_preds, val_labels)
print(f"Thresholds: {rounder.coef_}")
test_classes = rounder.predict(test_preds)
```

## Workflow

1. Train a regression model outputting continuous predictions
2. Generate predictions on the validation set
3. Optimize thresholds using Nelder-Mead to maximize QWK
4. Apply optimized thresholds to test predictions

## Key Decisions

- **Initial thresholds**: Start at midpoints (0.5, 1.5, ...) for classes 0,1,2,...
- **Sorting**: Sort optimized thresholds to ensure monotonicity
- **Metric**: QWK is most common; swap loss function for other ordinal metrics
- **Overfitting**: Thresholds can overfit small validation sets; use CV-averaged thresholds

## References

- [Intro APTOS Diabetic Retinopathy (EDA & Starter)](https://www.kaggle.com/code/tanlikesmath/intro-aptos-diabetic-retinopathy-eda-starter)
- [APTOS 2019 DenseNet Keras Starter](https://www.kaggle.com/code/xhlulu/aptos-2019-densenet-keras-starter)
