---
name: tabular-log-odds-fold-averaging
description: Average model predictions across CV folds in log-odds space rather than probability space for better-calibrated ensemble outputs
domain: tabular
---

# Log-Odds Fold Averaging

## Overview

When averaging predictions across CV folds, work in log-odds (logit) space instead of probability space. Log-odds averaging preserves the natural scale of the model's raw output and produces better-calibrated probabilities after inverse-logit transform. Especially effective for gradient boosting models that output raw log-odds by default.

## Quick Start

```python
import numpy as np

def log_odds_average(fold_predictions, from_proba=False):
    """Average predictions in log-odds space.
    
    Args:
        fold_predictions: list of arrays, one per fold
        from_proba: if True, convert probabilities to log-odds first
    """
    if from_proba:
        eps = 1e-7
        fold_predictions = [
            np.log(np.clip(p, eps, 1-eps) / (1 - np.clip(p, eps, 1-eps)))
            for p in fold_predictions
        ]
    
    avg_logits = np.mean(fold_predictions, axis=0)
    # Return raw log-odds (for ranking) or convert to probability
    return avg_logits  # or: 1 / (1 + np.exp(-avg_logits))

# With LightGBM: collect raw_score directly
fold_preds = []
for fold_model in models:
    raw = fold_model.predict(X_test, raw_score=True)  # log-odds
    fold_preds.append(raw)

final = log_odds_average(fold_preds)
```

## Key Decisions

- **raw_score=True**: LightGBM/XGBoost `predict(raw_score=True)` returns log-odds directly
- **Ranking tasks**: if only ranking matters, skip the sigmoid — raw log-odds preserve order
- **Probability tasks**: apply sigmoid after averaging for calibrated probabilities
- **Clipping**: when converting from probabilities, clip to [1e-7, 1-1e-7] to avoid inf

## References

- Source: [amex-lightgbm-quickstart](https://www.kaggle.com/code/ambrosm/amex-lightgbm-quickstart)
- Competition: American Express - Default Prediction
