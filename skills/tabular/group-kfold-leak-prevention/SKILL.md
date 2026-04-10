---
name: tabular-group-kfold-leak-prevention
description: >
  Uses GroupKFold to prevent data leakage when multiple rows share a common entity (e.g., same user, question, or document).
---
# GroupKFold Leak Prevention

## Overview

When rows in a dataset share a group identity (same question, user, session, etc.), standard KFold can place related rows in both train and validation, causing leakage. GroupKFold ensures all rows from a group land in the same fold, giving honest validation scores.

## Quick Start

```python
from sklearn.model_selection import GroupKFold

def group_kfold_split(X, y, groups, n_splits=5):
    """Generate leak-proof train/val splits by group.

    Args:
        X: features array
        y: target array
        groups: array of group IDs (e.g., question_id, user_id)
        n_splits: number of folds

    Yields:
        (train_idx, val_idx) tuples
    """
    gkf = GroupKFold(n_splits=n_splits)
    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        yield train_idx, val_idx
```

## Workflow

1. Identify the group column (entity that links related rows)
2. Replace `KFold` / `StratifiedKFold` with `GroupKFold`
3. Pass group IDs via the `groups` parameter
4. Train and evaluate per fold as usual
5. Average fold scores for final CV estimate

## Key Decisions

- **Group key selection**: Choose the entity that would cause leakage if split (question_body, user_id)
- **Stratification**: `GroupKFold` doesn't stratify; for imbalanced targets, use `StratifiedGroupKFold` (sklearn 1.0+)
- **Number of folds**: Must have at least `n_splits` unique groups
- **Nested groups**: If multiple group levels exist, group by the coarsest level

## References

- Google QUEST Q&A Labeling competition (Kaggle)
- Source: [quest-bert-base-tf2-0](https://www.kaggle.com/code/akensert/quest-bert-base-tf2-0)
