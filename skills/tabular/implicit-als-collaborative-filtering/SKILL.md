---
name: tabular-implicit-als-collaborative-filtering
description: >
  Alternating Least Squares matrix factorization on sparse user-item interaction matrices for implicit feedback recommendations.
---
# Implicit ALS Collaborative Filtering

## Overview

For implicit feedback (purchases, clicks — no ratings), factorize the sparse user-item interaction matrix into low-rank latent factors using ALS. The `implicit` library provides a GPU-accelerated implementation. Recommendations come from dot-product similarity between user and item factor vectors.

## Quick Start

```python
import implicit
import numpy as np
from scipy.sparse import coo_matrix

def build_interaction_matrix(df, user_col, item_col):
    users = df[user_col].astype("category")
    items = df[item_col].astype("category")
    data = np.ones(len(df))
    return coo_matrix((data, (users.cat.codes, items.cat.codes)))

coo = build_interaction_matrix(transactions, "customer_id", "article_id")
model = implicit.als.AlternatingLeastSquares(
    factors=100, iterations=15, regularization=0.01
)
model.fit(coo.tocsr())

# Recommend for a user
ids, scores = model.recommend(user_id, coo.tocsr()[user_id], N=12)
```

## Workflow

1. Build sparse COO matrix from transaction logs (user × item)
2. Convert to CSR format for efficient row slicing
3. Fit ALS model with tuned factors/iterations/regularization
4. Generate top-N recommendations per user via `model.recommend`
5. Batch inference in chunks (2000 users) for memory efficiency

## Key Decisions

- **Factors**: 50-200; higher captures more patterns but risks overfitting
- **Temporal filtering**: Train on recent window (7-21 days) to capture current trends
- **Validation**: MAP@K on a held-out recent time window, not random split
- **GPU**: `implicit` supports GPU; 10x faster for large matrices

## References

- H&M Personalized Fashion Recommendations (Kaggle)
- Source: [h-m-implicit-als-model-0-014](https://www.kaggle.com/code/julian3833/h-m-implicit-als-model-0-014)
