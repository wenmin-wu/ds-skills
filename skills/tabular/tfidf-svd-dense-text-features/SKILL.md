---
name: tabular-tfidf-svd-dense-text-features
description: Compress TF-IDF sparse text vectors into a handful of dense TruncatedSVD components so GBDTs can consume free-text fields as plain tabular columns
---

## Overview

Gradient boosters (LightGBM, XGBoost, CatBoost) do not consume sparse text matrices well — feature importance is diluted across thousands of rarely-hit columns, and training slows down. The clean fix is to fit a TF-IDF vectorizer on the union of train+test text, then run a small TruncatedSVD (3-20 components) and project each row into that dense subspace. You get a handful of columns like `svd_desc_1..svd_desc_5` that capture the dominant topic axes and drop straight into your tabular feature frame. Used on Avito Demand Prediction top kernels to fold title/description text into a LightGBM alongside price, category, and region.

## Quick Start

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd

tfidf = TfidfVectorizer(ngram_range=(1, 1), max_features=100_000)
full = tfidf.fit(pd.concat([train_df['description'], test_df['description']]).fillna(''))
train_vec = tfidf.transform(train_df['description'].fillna(''))
test_vec  = tfidf.transform(test_df['description'].fillna(''))

n_comp = 5
svd = TruncatedSVD(n_components=n_comp, algorithm='arpack', random_state=0)
svd.fit(train_vec)

cols = [f'svd_desc_{i+1}' for i in range(n_comp)]
train_df[cols] = svd.transform(train_vec)
test_df[cols]  = svd.transform(test_vec)
```

## Workflow

1. Fit one `TfidfVectorizer` on `train + test` text so the vocabulary is shared (transductive)
2. Transform train and test separately into sparse matrices
3. Fit `TruncatedSVD` with a small `n_components` on the train matrix
4. Project both splits into dense SVD columns with descriptive names per text field
5. Concat the dense block onto the tabular frame and feed the whole thing to the GBDT

## Key Decisions

- **Small n_components (3-10)**: more components add noise and training cost without helping GBDTs; they only need the topic axes.
- **Fit TF-IDF on train+test**: avoids OOV dimensionality mismatch. SVD itself is fit on train only to stay leakage-free.
- **`algorithm='arpack'`**: stable top-k decomposition on very sparse matrices; use `randomized` if n_components > 50.
- **One SVD per text field**: separate SVD blocks for title vs description outperform a single concat because the fields have different vocabularies.

## References

- [Simple Exploration + Baseline Notebook - Avito](https://www.kaggle.com/code/sudalairajkumar/simple-exploration-baseline-notebook-avito)
