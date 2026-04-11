---
name: tabular-prior-rebalancing-oversampling
description: >
  Rebalances training data by oversampling the majority class to match a known test-set class prior, reducing prediction miscalibration.
---
# Prior Rebalancing Oversampling

## Overview

When training data has a different class ratio than the test set (e.g., 37% positive in train vs 16.5% in test), models trained on the raw distribution produce miscalibrated probabilities. Instead of post-hoc calibration, resample the training set to match the known test prior. This is especially effective for log-loss metrics where calibration directly affects the score.

## Quick Start

```python
import pandas as pd
import numpy as np

test_prior = 0.165  # known or estimated test positive rate

pos = X_train[y_train == 1]
neg = X_train[y_train == 0]

# Scale negatives up to match test prior
scale = (len(pos) / (len(pos) + len(neg))) / test_prior - 1
neg_resampled = pd.concat([neg] * int(scale) + [neg[:int((scale % 1) * len(neg))]])

X_train = pd.concat([pos, neg_resampled]).sample(frac=1, random_state=42)
y_train = np.array([1] * len(pos) + [0] * len(neg_resampled))
```

## Workflow

1. Determine the test set class prior (from problem description or estimation)
2. Compute the resampling scale factor from train vs test prior ratio
3. Oversample the underrepresented class (relative to test prior)
4. Concatenate and shuffle
5. Train on the rebalanced dataset

## Key Decisions

- **Prior source**: Competition description, public LB probing, or domain knowledge
- **Over vs undersample**: Oversampling preserves all data; undersampling is faster
- **Post-hoc alternative**: Train on raw data, then calibrate with `CalibratedClassifierCV`
- **Metric sensitivity**: Most impactful for log-loss; less critical for AUC

## References

- [Data Analysis & XGBoost Starter (0.35460 LB)](https://www.kaggle.com/code/anokas/data-analysis-xgboost-starter-0-35460-lb)
