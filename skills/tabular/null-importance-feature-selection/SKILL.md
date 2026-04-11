---
name: tabular-null-importance-feature-selection
description: >
  Scores features by comparing actual importances against a null distribution from shuffled targets, removing features that cannot beat random noise.
---
# Null Importance Feature Selection

## Overview

Standard feature importance (gain/split) reflects how useful a feature is for the model but doesn't account for noise features that appear useful by chance. Null importance builds a baseline: train the model many times with shuffled targets to create a distribution of "random" importances, then keep only features whose actual importance significantly exceeds their null distribution. Typically improves CV by 0.001–0.005 and reduces overfitting on noisy wide datasets.

## Quick Start

```python
import lightgbm as lgb
import numpy as np
import pandas as pd

def get_feature_importances(data, features, target, shuffle=False):
    y = target.copy()
    if shuffle:
        y = y.sample(frac=1.0).reset_index(drop=True)
    dtrain = lgb.Dataset(data[features], y, free_raw_data=False)
    clf = lgb.train(
        {'objective': 'binary', 'boosting_type': 'rf',
         'subsample': 0.623, 'colsample_bytree': 0.7,
         'num_leaves': 127, 'max_depth': 8, 'bagging_freq': 1},
        dtrain, num_boost_round=200
    )
    imp = pd.DataFrame({'feature': features,
                        'importance': clf.feature_importance(importance_type='gain')})
    return imp

# Actual importances
actual_imp = get_feature_importances(df, feats, df['target'], shuffle=False)

# Null distribution (80 runs)
null_imp = pd.concat([
    get_feature_importances(df, feats, df['target'], shuffle=True)
    for _ in range(80)
])

# Score: % of null runs where importance < 25th pctl of actual
scores = []
for f in feats:
    f_null = null_imp.loc[null_imp['feature'] == f, 'importance'].values
    f_act = actual_imp.loc[actual_imp['feature'] == f, 'importance'].values
    score = 100 * (f_null < np.percentile(f_act, 25)).sum() / len(f_null)
    scores.append((f, score))

selected = [f for f, s in scores if s >= 80]
```

## Workflow

1. Train model on real targets → record actual feature importances
2. Repeat N times (50–100) with shuffled target → build null importance distribution
3. Score each feature: how often does actual importance exceed null baseline
4. Sweep score thresholds (e.g., 60, 70, 80, 90) and evaluate CV at each cutoff
5. Select threshold that maximizes CV metric

## Key Decisions

- **N shuffles**: 80 is a good default; fewer (50) for speed, more (200) for stability
- **Importance type**: `gain` is more discriminative than `split` for this method
- **Scoring**: log-ratio or percentile-based — percentile is more interpretable
- **Model**: Use RF mode (`boosting_type='rf'`) for faster, less correlated runs

## References

- [Feature Selection with Null Importances](https://www.kaggle.com/code/ogrellier/feature-selection-with-null-importances)
