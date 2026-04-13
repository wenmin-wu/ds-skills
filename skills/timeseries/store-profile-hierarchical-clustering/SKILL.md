---
name: timeseries-store-profile-hierarchical-clustering
description: Re-cluster retail stores by scale-normalized weekday/dayoff mean+std profiles using Ward agglomerative clustering, replacing vendor-supplied "type/cluster" labels that correlate with store size instead of demand shape
---

## Overview

Vendor-supplied store `type` and `cluster` labels (common in retail datasets like Favorita) usually encode store size, not demand *shape*. A store profile is the right grouping key when you plan to train per-group models or apply per-group post-processing: two small stores with identical weekday shapes should cluster together even if one sells twice as much. The recipe is three steps — normalize each row by its store's overall mean to kill scale, aggregate to a per-store weekday mean+std profile, then Ward-cluster the profiles.

## Quick Start

```python
import pandas as pd
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.cluster import AgglomerativeClustering

store_avg = sales.groupby('store_nbr')['transactions'].mean()
sales['normalized'] = sales['transactions'] / sales['store_nbr'].map(store_avg)

profile = (sales.groupby(['store_nbr', 'day_of_week'])['normalized']
                .agg(['mean', 'std'])
                .unstack(level='day_of_week'))   # (n_stores, 14)

dendrogram(ward(profile.values))                 # eyeball a cutoff k
labels = AgglomerativeClustering(n_clusters=6).fit_predict(profile.values)
stores['shape_cluster'] = labels
```

## Workflow

1. Compute each store's overall mean target and divide every row by it → scale-invariant normalized target
2. Group by `(store, day_of_week)` and aggregate `mean` **and** `std` → unstack so each store is one feature row (7 × 2 = 14 cols)
3. Optionally add a second unstack by a `dayoff` flag for 28-dim profiles that separate holiday behavior
4. Plot a Ward dendrogram on the profile matrix and pick `k` at the biggest vertical gap
5. Fit `AgglomerativeClustering(k)` on the profile matrix and attach labels to the stores table
6. Use `shape_cluster` as a grouping key for per-cluster models or as a categorical feature in a global model

## Key Decisions

- **Normalize before aggregating**: raw means are dominated by store size and destroy shape clustering.
- **Mean + std, not just mean**: two stores with equal weekday averages but different volatility behave differently on promo days.
- **Ward linkage**: produces compact equal-variance clusters well-matched to mean/std features; single-linkage produces chains.
- **Pick k from dendrogram gap**: silhouette is unreliable on 14-dim profiles; visual inspection is faster and more interpretable.
- **Discard vendor labels**: if the dendrogram shows clear shape groups that don't match the vendor's type/cluster, the vendor labels are noise.

## References

- [A first Kaggle - Part 1 - Forecasting store #47](https://www.kaggle.com/code/jagangupta/a-first-kaggle-part-1-forecasting-store-47)
