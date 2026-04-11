---
name: tabular-logit-transform-stacking
description: >
  Applies logit transformation to base model probabilities before fitting a logistic regression meta-learner, enabling principled linear combination in log-odds space.
---
# Logit Transform Stacking

## Overview

When stacking binary classification models, the base model outputs are probabilities in [0, 1]. Fitting a linear meta-learner directly on probabilities is suboptimal because the relationship between probabilities and log-odds is nonlinear — a 0.01 difference near 0.5 means less than near 0.99. Applying logit (`log(p/(1-p))`) transforms probabilities to log-odds space where linear combination is theoretically justified. The logistic regression meta-learner then learns optimal weights in this space, often outperforming probability-space blending by 0.001–0.003 AUC.

## Quick Start

```python
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from sklearn.linear_model import LogisticRegression

ALMOST_ZERO = 1e-10
ALMOST_ONE = 1 - ALMOST_ZERO

# Load OOF predictions from base models
oof_preds = {}
for name, path in base_model_files.items():
    probs = pd.read_csv(path)['prediction']
    oof_preds[name] = logit(probs.clip(ALMOST_ZERO, ALMOST_ONE))

X_train = pd.DataFrame(oof_preds).values
y_train = true_labels

# Fit meta-learner in logit space
meta = LogisticRegression(C=1.0)
meta.fit(X_train, y_train)

# Inspect learned weights
weights = meta.coef_[0] / meta.coef_[0].sum()
print("Model weights:", dict(zip(oof_preds.keys(), weights)))

# Apply to test predictions
test_preds = {}
for name, path in test_model_files.items():
    probs = pd.read_csv(path)['prediction']
    test_preds[name] = logit(probs.clip(ALMOST_ZERO, ALMOST_ONE))

X_test = pd.DataFrame(test_preds).values
final_probs = meta.predict_proba(X_test)[:, 1]
```

## Workflow

1. Generate OOF predictions from each base model
2. Clip probabilities to `[1e-10, 1-1e-10]` to avoid infinite logit values
3. Apply `logit()` transform to all base model outputs
4. Fit `LogisticRegression` on logit-transformed OOF predictions
5. Transform test predictions the same way, then predict with meta-learner

## Key Decisions

- **Clipping**: Essential — logit(0) = -inf, logit(1) = inf. Clip to `[1e-10, 1-1e-10]`
- **Regularization**: C=1.0 default; lower C for many correlated base models
- **vs probability blending**: Logit stacking is strictly better when base models vary in calibration
- **Weight interpretation**: Normalized coefficients ≈ blending weights, but in log-odds space

## References

- [Simple Linear Stacking (LB .9730)](https://www.kaggle.com/code/aharless/simple-linear-stacking-lb-9730)
