---
name: tabular-haversine-knn-candidate-generation
description: >
  Generates geographically proximate candidate pairs for entity matching using KNN with haversine distance, optionally partitioned by country.
---
# Haversine KNN Candidate Generation

## Overview

Entity matching at scale requires a candidate generation step — comparing all N^2 pairs is infeasible. For location-based entities (POIs, stores, addresses), KNN with haversine distance finds the K geographically closest candidates per record. Partitioning by country/region reduces the search space further and prevents cross-continent false matches. The resulting candidate pairs are then scored by a downstream classifier.

## Quick Start

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

def generate_geo_candidates(df, n_neighbors=20, partition_col="country"):
    """Generate candidate pairs using haversine KNN per partition."""
    all_candidates = []
    for group, group_df in df.groupby(partition_col):
        group_df = group_df.reset_index(drop=True)
        coords = np.deg2rad(group_df[["latitude", "longitude"]].values)
        knn = NearestNeighbors(
            n_neighbors=min(len(group_df), n_neighbors),
            metric="haversine", n_jobs=-1
        )
        knn.fit(coords)
        dists, indices = knn.kneighbors(coords)
        for i in range(len(group_df)):
            for j in range(1, len(indices[i])):  # skip self-match
                all_candidates.append({
                    "id": group_df.iloc[i]["id"],
                    "match_id": group_df.iloc[indices[i][j]]["id"],
                    "geo_dist": dists[i][j] * 6371,  # km
                    "neighbor_rank": j,
                })
    return pd.DataFrame(all_candidates)

candidates = generate_geo_candidates(df, n_neighbors=20)
```

## Workflow

1. Convert lat/lon to radians (required by haversine metric)
2. Partition data by country or region to reduce search space
3. Fit KNN with haversine metric per partition
4. Extract K nearest neighbors and distances for each record
5. Build candidate pair DataFrame with distance and rank features

## Key Decisions

- **n_neighbors**: 10-50 typical; higher recall but more pairs to classify
- **Partitioning**: By country prevents cross-region false matches; skip for global matching
- **Distance unit**: Haversine returns radians; multiply by 6371 for kilometers
- **Dual index**: Combine geo KNN with text-based KNN for higher recall

## References

- [Foursquare - LightGBM Baseline](https://www.kaggle.com/code/ryotayoshinobu/foursquare-lightgbm-baseline)
- [Public: 0.861 | PyKakasi & Radian Coordinates](https://www.kaggle.com/code/nlztrk/public-0-861-pykakasi-radian-coordinates)
