---
name: tabular-outlier-aware-two-stage-blending
description: When a regression target has a long discrete tail (e.g. ~1% of rows pinned at -33.22 in Elo), train one regressor on the *non-outlier* subset, a separate binary classifier for the outlier flag, and splice the predictions — replace the top-K most-confident outlier predictions in the regressor's output with the outlier value, where K is calibrated on validation
---

## Overview

The Elo Merchant target had ~1% of rows fixed at -33.21928 (a hard "loyalty churn" sentinel). A single regressor was forced to compromise: either it learned to predict the sentinel and damaged its non-outlier RMSE, or it ignored it and lost easy points. Every top solution decoupled the problem: (A) a regressor trained *with outliers removed* — clean, accurate on the bulk; (B) a binary classifier `is_outlier` over all rows; then a *splice*: take the regressor's full-test predictions, sort the classifier scores, and overwrite the top-K predictions with the sentinel value. K is chosen by minimizing RMSE on the validation fold, typically around `K ≈ 1.05 × (val_outlier_count)`. This pattern beat single-model baselines by 0.01-0.02 RMSE on the public LB.

## Quick Start

```python
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb

OUT_VAL = -33.21928

# A) regressor on non-outliers
mask = train.target != OUT_VAL
reg = lgb.LGBMRegressor(**reg_params).fit(train.loc[mask, feats], train.loc[mask, 'target'])
test_reg = reg.predict(test[feats])

# B) outlier classifier on all rows
clf = lgb.LGBMClassifier(**clf_params).fit(
    train[feats], (train.target == OUT_VAL).astype(int))
test_out_score = clf.predict_proba(test[feats])[:, 1]

# Splice: replace top-K by classifier score
K = int(len(test) * 0.011)  # tuned on validation
top_idx = np.argsort(-test_out_score)[:K]
final = test_reg.copy()
final[top_idx] = OUT_VAL
```

## Workflow

1. Detect the discrete sentinel — histogram the target, look for a spike at a single value
2. Train a clean regressor with `target != sentinel` rows only — RMSE should drop substantially vs. the full-train baseline
3. Train a binary classifier on all rows for `is_sentinel`
4. On validation: sort by classifier score and try K ∈ {0.9·n_out, ..., 1.2·n_out}, pick the K that minimizes RMSE
5. Apply the same K-fraction at test time
6. Optional: blend the spliced output with a single-model baseline at e.g. `0.6 × spliced + 0.4 × full` for variance reduction

## Key Decisions

- **Splice, don't sum**: an additive `regressor + classifier × sentinel_offset` underflows because the sentinel is so far from the bulk; a hard overwrite is cleaner.
- **K is the only tunable knob**: get it from validation, not from a fixed percentile.
- **Classifier is heavily imbalanced**: use `class_weight='balanced'` or downsample non-outliers before fitting.
- **Don't oversample the regressor**: removing outliers is the entire point — putting them back via SMOTE defeats it.
- **Generalizes beyond Elo**: any target with a discrete sentinel (refund of -100%, capped score of 5.0, default flag) fits this pattern.
- **Cross-validate the splice K**: a K tuned on a single fold is unstable; average across 5 folds.

## References

- [Combining your model with a model without outlier — Elo Merchant](https://www.kaggle.com/competitions/elo-merchant-category-recommendation)
