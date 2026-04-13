---
name: timeseries-rolling-refit-arima-forecast
description: Walk-forward validation for ARIMA by refitting on history at each step, forecasting one step ahead, then appending the true observation — produces an honest one-step error distribution that mirrors nightly-retrained production forecasters
---

## Overview

`ARIMA.forecast(steps=h)` compounds error and hides refit cost. A rolling one-step loop — refit on current history, predict the next point, append the *true* value — gives you the error distribution a nightly-retrained production model would actually see. The distinction matters: multi-step ARIMA looks much worse than it really is because errors integrate, and a static-fit model looks much better than it really is because it never sees drift. The rolling refit is the honest middle ground.

## Quick Start

```python
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

history = list(train)
predictions = []
for t in range(len(test)):
    fit = ARIMA(history, order=(5, 0, 5)).fit(method='css')  # css = faster
    yhat = float(fit.forecast(steps=1)[0])
    predictions.append(yhat)
    history.append(test.iloc[t])          # append TRUTH, not prediction

pred = pd.Series(predictions, index=test.index)
abs_err = (test - pred).abs()
```

## Workflow

1. Split the series into train/test by a cutoff date
2. Copy `train` into a mutable `history` list
3. For each test timestamp: refit ARIMA on `history`, forecast one step, record it
4. Append the observed value (never the prediction) to `history` before the next iteration
5. Assemble predictions into a Series aligned to `test.index`; plot observed vs forecast and the absolute-error distribution

## Key Decisions

- **Append truth, not prediction**: otherwise errors compound and you're back to the broken multi-step case.
- **Refit every step**: the whole point is to catch drift. A static fit defeats the pattern.
- **`method='css'`**: conditional sum of squares is faster than full MLE and plenty accurate for walk-forward. Bump `maxiter` if (p,q) are large.
- **Pick (p,q) once via `arma_order_select_ic`**: BIC over a small grid on the train split, then freeze. Re-selecting each step wastes compute and rarely changes the order.
- **Cost is O(len(test))**: budget accordingly; for 10k+ test points, switch to state-space Kalman updates instead of full refit.

## References

- [A first Kaggle - Part 1 - Forecasting store #47](https://www.kaggle.com/code/jagangupta/a-first-kaggle-part-1-forecasting-store-47)
