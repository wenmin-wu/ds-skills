---
name: tabular-content-difficulty-features
description: >
  Precomputes item/content difficulty as historical mean accuracy, merged as a static feature for user-item prediction tasks.
---
# Content Difficulty Features

## Overview

For recommendation or knowledge tracing tasks, compute each item's historical difficulty (mean success rate) from training data and merge as a static feature. This gives the model a strong prior: some questions are inherently harder. Works for any user-item interaction dataset.

## Quick Start

```python
import pandas as pd

def add_content_features(train_df, content_col='content_id', target_col='answered_correctly'):
    """Compute and merge content-level difficulty features."""
    content_stats = (
        train_df.groupby(content_col)[target_col]
        .agg(['mean', 'count', 'std'])
        .reset_index()
    )
    content_stats.columns = [content_col, 'content_mean', 'content_count', 'content_std']

    # Fill NaN std for items with single interaction
    content_stats['content_std'] = content_stats['content_std'].fillna(0)

    return train_df.merge(content_stats, on=content_col, how='left')

# Usage
train = add_content_features(train, 'content_id', 'answered_correctly')
# For test: merge same content_stats (computed from train only)
```

## Workflow

1. Group training data by content/item ID
2. Compute mean (difficulty), count (popularity), std (consistency)
3. Merge back as static features via left join
4. For test data, use the same content_stats from training
5. Handle unseen items with global mean imputation

## Key Decisions

- **Train-only stats**: Never include test outcomes in difficulty computation
- **Smoothing**: For rare items, blend with global mean: `(count*mean + prior*global) / (count + prior)`
- **Multiple levels**: Compute difficulty per content, per tag, per bundle for different granularities
- **Temporal decay**: Optionally weight recent interactions higher for drifting difficulty

## References

- Riiid Answer Correctness Prediction (Kaggle)
- Source: [riiid-comprehensive-eda-baseline](https://www.kaggle.com/code/erikbruin/riiid-comprehensive-eda-baseline)
