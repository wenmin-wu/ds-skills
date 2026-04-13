---
name: timeseries-tweedie-objective-zero-inflated
description: Use LightGBM's tweedie objective with variance_power between 1.05 and 1.2 for zero-inflated count forecasting (retail SKUs, intermittent demand, click events) — handles the "many zeros plus a heavy right tail" distribution that breaks both regression (RMSE) and classification (BCE) objectives
---

## Overview

Retail demand at the SKU level is brutally zero-inflated: 60-90% of (item, store, day) combinations sell zero units, the non-zero values follow a long-tailed positive distribution, and the conditional mean varies by orders of magnitude across SKUs. RMSE pretends the noise is Gaussian and produces over-smoothed forecasts that miss every spike. Poisson is closer but its variance-equals-mean assumption is too restrictive. The Tweedie distribution interpolates: at `variance_power=1.0` it's Poisson, at `variance_power=2.0` it's Gamma, in between (1.05-1.2) it's a compound Poisson-Gamma that explicitly models a positive probability mass at 0 with a Gamma-distributed positive tail. LightGBM has native support — one objective change buys you 5-10% WRMSSE improvement on intermittent demand without any other code changes.

## Quick Start

```python
import lightgbm as lgb

params = {
    'objective': 'tweedie',
    'tweedie_variance_power': 1.1,      # 1.0=Poisson, 2.0=Gamma; 1.1 is canonical for retail
    'metric': 'rmse',
    'num_leaves': 2**11 - 1,
    'min_data_in_leaf': 2**12 - 1,
    'feature_fraction': 0.5,
    'max_bin': 100,
    'boost_from_average': False,        # critical for Tweedie with sparse positives
    'learning_rate': 0.03,
    'num_iterations': 1400,
}

train_data = lgb.Dataset(X_train, label=y_train)
model = lgb.train(params, train_data, valid_sets=[lgb.Dataset(X_val, label=y_val)])

# predictions are non-negative by construction
preds = model.predict(X_test)
```

## Workflow

1. Verify the target is non-negative and zero-inflated (>30% zeros, positive long tail)
2. Set `objective='tweedie'` and tune `tweedie_variance_power` over `[1.05, 1.1, 1.15, 1.2]`
3. Set `boost_from_average=False` — the default `True` initializes from the global mean, which biases sparse positives away from zero
4. Use `metric='rmse'` for monitoring even though training is Tweedie — Tweedie deviance is hard to interpret
5. Apply a small post-hoc bias correction (`* 1.02`) if validation predictions sum below the validation actuals
6. Check the predicted distribution: if it's still all near-zero, lower the variance_power; if it has too many big spikes, raise it

## Key Decisions

- **`tweedie_variance_power=1.1`**: canonical default for retail. Lower means more like Poisson (good for very sparse), higher means more like Gamma (good for less-sparse-but-heavy-tail).
- **`boost_from_average=False`**: Tweedie's link function makes the global average a poor starting point on zero-inflated data; starting from zero converges faster.
- **Don't log-transform the target**: Tweedie already handles the heavy tail; log + Tweedie double-corrects and underpredicts the spikes.
- **vs. Poisson**: Poisson works for fully sparse counts but underestimates variance on the positive tail.
- **vs. zero-inflated Poisson hurdle**: hurdle models need two networks and sklearn integration; Tweedie is one LightGBM call.
- **`max_bin=100`**: lower than default helps Tweedie because the large `min_data_in_leaf` already prevents over-splitting.

## References

- [M5 - Three shades of Dark: Darker magic](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
