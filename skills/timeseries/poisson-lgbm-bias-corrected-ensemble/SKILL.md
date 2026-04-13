---
name: timeseries-poisson-lgbm-bias-corrected-ensemble
description: Train a single Poisson LightGBM count forecaster, then ensemble its predictions with multiple multiplicative scaling factors (alpha ≈ 1.02-1.03) to undo the systematic downward bias of Poisson regression on intermittent retail data
---

## Overview

Poisson regression LightGBM is the right objective for non-negative count forecasting (sales, clicks, visits) because it naturally produces non-negative predictions and handles the count distribution. But it has a known systematic flaw: on intermittent / zero-inflated data, the trained model consistently predicts means a few percent below the true conditional mean, because the loss surface around zero is asymmetric and the optimizer settles into a slightly conservative basin. The fix is embarrassingly simple: train one model, then "ensemble" it with itself by applying a small set of multiplicative scaling factors, one per ensemble member. Average their predictions and you get a bias-corrected forecast at zero extra training cost.

## Quick Start

```python
import lightgbm as lgb
import numpy as np

params = {
    'objective': 'poisson',
    'metric': 'rmse',
    'learning_rate': 0.075,
    'sub_row': 0.75,
    'bagging_freq': 1,
    'lambda_l2': 0.1,
    'num_leaves': 128,
    'min_data_in_leaf': 100,
    'num_iterations': 1200,
}

model = lgb.train(params, train_data, valid_sets=[valid_data])

base = model.predict(X_test)
alphas = [1.018, 1.023, 1.028]              # tuned on validation
preds = np.mean([a * base for a in alphas], axis=0)
```

## Workflow

1. Train the Poisson model normally with reasonable hyperparameters
2. Predict on validation and compute the ratio `actual.sum() / pred.sum()` — typically 1.02-1.04
3. Pick three multipliers bracketing that ratio (e.g. `[1.018, 1.023, 1.028]` for a center of 1.023)
4. At test time, predict once and apply each multiplier; average the resulting forecasts
5. Validate that the aggregate sum of corrected predictions matches the historical aggregate within 1-2%

## Key Decisions

- **Three multipliers around the validation ratio, not one**: a single multiplier is brittle to small validation noise; three give variance reduction without retraining.
- **Multiplicative, not additive**: additive correction would lift zero predictions above zero, polluting the count semantics. Multiplicative leaves zeros alone.
- **No retraining needed**: this is a post-hoc calibration; the same model can be reused for many alpha sweeps.
- **vs. Tweedie**: Tweedie handles the bias internally if `variance_power < 1.2`; if you're already on Tweedie, you don't need this trick.
- **Tune alpha on validation, not test**: extrapolating from one fold can over-correct; use a held-out set.
- **Don't apply this to RMSE-trained models**: RMSE has different asymmetry — the multiplier sign would be wrong.

## References

- [M5 First Public Notebook Under 0.50](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
