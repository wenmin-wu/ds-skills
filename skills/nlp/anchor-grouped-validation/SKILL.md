---
name: nlp-anchor-grouped-validation
description: Split validation by unique anchor/query entities so no anchor appears in both train and val, preventing data leakage in pairwise matching tasks
---

# Anchor-Grouped Validation

## Overview

In pairwise matching tasks (phrase similarity, question-answer matching), the same anchor/query maps to multiple targets. If the same anchor appears in both train and validation, the model memorizes anchor-specific patterns rather than learning general similarity. Splitting by unique anchors ensures the model is evaluated on truly unseen queries.

## Quick Start

```python
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

groups = df['anchor'].values
scores = (df['score'] * 100).astype(int)  # bin for stratification

sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, scores, groups)):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    assert set(train_df['anchor']) & set(val_df['anchor']) == set()
```

## Workflow

1. Identify the grouping entity (anchor, query, user, document ID)
2. Use `StratifiedGroupKFold` for balanced + grouped splits, or manual shuffle-split on unique entities
3. Verify no group overlap between train and validation sets
4. Report both in-group (standard) and grouped CV scores — the gap reveals leakage risk

## Key Decisions

- **StratifiedGroupKFold vs GroupKFold**: stratified variant preserves label distribution across folds
- **Stratification target**: bin continuous scores into integers for stratification compatibility
- **Number of folds**: 4-5 is standard; fewer folds if number of unique anchors is small
- **Simple alternative**: shuffle unique anchors, split first 25% as val, rest as train

## References

- [Iterate like a grandmaster!](https://www.kaggle.com/code/jhoward/iterate-like-a-grandmaster)
