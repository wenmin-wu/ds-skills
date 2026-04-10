---
name: cv-knn-distance-threshold-matching
description: KNN-based retrieval with grid-searched distance threshold to convert embedding neighbors into match predictions
domain: cv
---

# KNN Distance Threshold Matching

## Overview

After extracting embeddings (image or text), find candidate matches via K-nearest neighbors, then apply a distance threshold to decide which neighbors are true matches. Grid-search the threshold on a validation set using F1 score. Optionally filter outliers using z-score on the distance distribution per query.

## Quick Start

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors  # or cuml.neighbors

def find_matches(embeddings, ids, threshold, k=50):
    """Find matching items via KNN + distance threshold.
    
    Args:
        embeddings: (N, D) normalized feature vectors
        ids: array of item identifiers
        threshold: max distance to consider a match
        k: number of neighbors to retrieve
    Returns:
        list of matched ID arrays per query
    """
    knn = NearestNeighbors(n_neighbors=min(k, len(embeddings)), metric='cosine')
    knn.fit(embeddings)
    distances, indices = knn.kneighbors(embeddings)
    
    predictions = []
    for i in range(len(embeddings)):
        mask = distances[i] < threshold
        predictions.append(ids[indices[i][mask]])
    return predictions

# Grid search optimal threshold
best_score, best_thresh = 0, 0
for thresh in np.arange(0.1, 1.0, 0.05):
    preds = find_matches(val_embeddings, val_ids, thresh)
    score = compute_f1(val_labels, preds)
    if score > best_score:
        best_score, best_thresh = score, thresh
```

## Key Decisions

- **Cosine vs L2**: cosine distance for normalized embeddings; L2 if unnormalized
- **K=50**: retrieve more neighbors than expected matches — threshold handles filtering
- **Grid search step**: 0.05 is a good start; narrow to 0.01 around the best range
- **Per-query outlier filtering**: use z-score on distances to adaptively tighten threshold

## References

- Source: [unsupervised-baseline-arcface](https://www.kaggle.com/code/ragnar123/unsupervised-baseline-arcface)
- Competition: Shopee - Price Match Guarantee
