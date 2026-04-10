---
name: cv-multimodal-prediction-union
description: Combine match predictions from image embeddings, text similarity, and perceptual hash via set union for maximum recall
domain: cv
---

# Multimodal Prediction Union

## Overview

In product/image matching, different signals (CNN embeddings, text TF-IDF, perceptual hash) each catch different true matches. Take the set union of all per-signal predictions to maximize recall. This is simpler and often better than learned fusion for retrieval tasks where precision can be traded for recall.

## Quick Start

```python
import numpy as np
import pandas as pd

def union_predictions(df, pred_columns):
    """Merge predictions from multiple signals via set union.
    
    Args:
        df: DataFrame where each pred_column contains arrays of matched IDs
        pred_columns: list of column names with per-signal match arrays
    Returns:
        Series of unique merged match arrays
    """
    def merge_row(row):
        all_ids = np.concatenate([row[col] for col in pred_columns])
        return np.unique(all_ids)
    return df.apply(merge_row, axis=1)

# Usage: each column has arrays of matched item IDs
df['image_matches'] = find_matches(image_embeddings, ids, img_thresh)
df['text_matches'] = find_matches(text_embeddings, ids, txt_thresh)
df['hash_matches'] = phash_group_matches(df)

df['final_matches'] = union_predictions(
    df, ['image_matches', 'text_matches', 'hash_matches']
)
```

## Key Decisions

- **Union over intersection**: maximizes recall at slight precision cost — appropriate for retrieval
- **Per-signal thresholds**: tune each modality's threshold independently before merging
- **Order doesn't matter**: set union is commutative — no need to prioritize signals
- **Diminishing returns**: typically 3-4 signals saturate; more signals add noise without recall gain

## References

- Source: [part-2-rapids-tfidfvectorizer-cv-0-700](https://www.kaggle.com/code/cdeotte/part-2-rapids-tfidfvectorizer-cv-0-700)
- Competition: Shopee - Price Match Guarantee
