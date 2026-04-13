---
name: tabular-sparse-dense-hstack-lgbm
description: Train LightGBM directly on a scipy.sparse.hstack of TF-IDF text vectors and dense tabular columns, passing feature_name and categorical_feature so native categorical handling survives the sparse block
---

## Overview

When you have both wide sparse text features (TF-IDF, CountVectorizer — 50k to 500k columns) and dense tabular columns, the wrong move is to densify the sparse block (OOM) or to SVD-compress it (loses signal). LightGBM accepts CSR matrices natively, so you can `scipy.sparse.hstack` the text block and the dense predictors into one CSR matrix and hand it straight to `lgb.Dataset`. The trick is preserving feature names and the categorical-feature flag across the boundary, so LightGBM still uses its native categorical splits on the label-encoded columns living inside the otherwise-sparse matrix.

## Quick Start

```python
import scipy.sparse as sp
import numpy as np
import lightgbm as lgb

X_train = sp.hstack([
    train_desc_tfidf,       # (n, ~100k) sparse
    train_title_counts,     # (n, ~20k) sparse
    train[dense_cols].values,  # (n, d) dense
], format='csr')

feature_names = np.hstack([
    tfidf.get_feature_names_out(),
    cv.get_feature_names_out(),
    dense_cols,
]).tolist()

dtrain = lgb.Dataset(
    X_train, label=y_train,
    feature_name=feature_names,
    categorical_feature=categorical_cols,
)

params = dict(objective='regression', metric='rmse',
              num_leaves=32, max_depth=15, learning_rate=0.02,
              feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=5)

model = lgb.train(params, dtrain, valid_sets=[dtrain, dvalid],
                  num_boost_round=16000, early_stopping_rounds=500)
```

## Workflow

1. Fit each text vectorizer on `train + test` concatenated to share vocabulary
2. Label-encode categorical columns with `fit` on the train+test union (avoids unseen labels)
3. `sp.hstack([sparse_blocks..., dense.values], format='csr')` — CSR is required by LightGBM
4. Build the aligned `feature_name` list by concatenating `get_feature_names_out()` from each block and the dense column names
5. Pass both `feature_name` and `categorical_feature` (as a list of names) to `lgb.Dataset` so native categorical splits still work
6. Train with a low `feature_fraction` (0.3-0.6) to subsample the wide feature space each tree

## Key Decisions

- **CSR, not CSC or COO**: LightGBM converts internally but errors on other layouts. `format='csr'` is mandatory.
- **`feature_fraction` ≤ 0.6**: with 100k+ columns, unrestricted trees slow to a crawl and overfit — subsample every split.
- **Pass categoricals by name, not index**: indices change if the block widths change; names are stable.
- **Fit encoders/vectorizers on train+test union**: unseen categorical labels or TF-IDF tokens at inference break the column alignment.
- **vs. SVD-compressing the text**: SVD loses rare-token signal. Sparse hstack keeps everything and lets the tree choose.
- **Clip predictions to target range**: `np.clip(preds, 0, 1)` for bounded targets like deal probability — LightGBM regression is unconstrained.

## References

- [Aggregated features & LightGBM](https://www.kaggle.com/code/bminixhofer/aggregated-features-lightgbm)
