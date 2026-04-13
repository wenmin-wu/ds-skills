---
name: timeseries-recursive-multistep-forecasting
description: Forecast a multi-step horizon by predicting one day ahead, writing the prediction back into the panel as the new "actual", recomputing all lag and rolling features that depend on it, then predicting the next day — turns a one-step LightGBM regressor into a 28-day forecaster without changing the model
---

## Overview

Direct multi-step forecasting (one model per horizon day) is expensive: 28 days = 28 models. Recursive forecasting trains one one-step model and reuses it for every horizon day by feeding its own predictions back as inputs. The trick is that lag features (`sales.shift(7)`) and rolling features (`sales.rolling(28).mean()`) at horizon day H+k depend on predictions from days H..H+k-1, so you must recompute them *inside* the prediction loop, not once before. Done correctly, you get 28-day forecasts with the same model that scored well on 1-step validation. Done wrong (precomputed features that don't see the predictions), you get garbage.

## Quick Start

```python
import pandas as pd
from datetime import timedelta

base_test = build_panel_with_unknown_target()   # rows for entire horizon, sales=NaN

for h in range(1, 29):                          # 28-day horizon
    day = first_forecast_day + timedelta(days=h - 1)

    # window includes max_lags days before today so we can recompute features
    window = base_test[
        (base_test.date >= day - timedelta(days=max_lags)) &
        (base_test.date <= day)
    ].copy()

    create_features(window)                     # lags + rollings using up-to-day data

    today = window.loc[window.date == day, train_cols]
    yhat = alpha * model.predict(today)         # alpha = bias-correction multiplier

    base_test.loc[base_test.date == day, 'sales'] = yhat
```

## Workflow

1. Build a panel that already contains the future horizon rows with `sales=NaN`
2. Loop `h` over the horizon days
3. Slice a sliding window covering `[day - max_lags, day]` so feature creation has enough history
4. Recompute lag and rolling features on the window — this is the step everyone misses
5. Predict for the current day, optionally apply a bias-correction multiplier (`alpha ≈ 1.02-1.03` for Poisson)
6. Write the prediction back into `base_test` so the next iteration's features see it
7. After the loop, the entire horizon column is filled

## Key Decisions

- **Recompute features inside the loop**: precomputing lag-7 once before the loop means day 8's "lag-7" is actually a previously-predicted day-1 — but you need that prediction to exist *before* you compute it. The only correct order is predict → write → recompute → predict.
- **Window slice for speed**: don't recompute features on the entire panel — just on `[day - max_lags, day]`. The rest is irrelevant for today's prediction.
- **Bias-correction multiplier `alpha`**: Poisson and Tweedie LightGBM consistently underpredict by 2-3%; multiply by ~1.02-1.03 (tuned on validation) before writing back, otherwise the bias compounds across the horizon.
- **Single-store batches**: if you train per-store models, group by store inside the loop to amortize the model load.
- **vs. direct multi-output**: direct is more accurate for short horizons, recursive scales better and uses one model.

## References

- [M5 First Public Notebook Under 0.50](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
- [M5 - Three shades of Dark: Darker magic](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
