---
name: tabular-majority-vote-submission-blend
description: Blend multiple submission CSVs by row-wise majority voting on discrete predictions to produce a more robust final output
---

# Majority Vote Submission Blend

## Overview

When you have multiple independently trained pipelines producing discrete class predictions, majority voting is the simplest and most robust blending strategy. Unlike rank averaging or weighted blending (which require continuous scores), majority voting works directly on integer labels and is especially effective for ordinal classification where averaging can land between classes.

## Quick Start

```python
import pandas as pd

def majority_vote_blend(submission_paths, id_col='id', target_col='target'):
    subs = [pd.read_csv(p).sort_values(id_col).reset_index(drop=True)
            for p in submission_paths]

    combined = pd.DataFrame({id_col: subs[0][id_col]})
    for i, s in enumerate(subs):
        combined[f'pred_{i}'] = s[target_col].values

    pred_cols = [c for c in combined.columns if c.startswith('pred_')]
    combined[target_col] = (combined[pred_cols]
        .mode(axis=1)[0].astype(int))

    return combined[[id_col, target_col]]

final = majority_vote_blend([
    'sub_lgbm.csv', 'sub_xgb.csv', 'sub_catboost.csv'
])
final.to_csv('submission.csv', index=False)
```

## Workflow

1. Generate N submission CSVs from independent model pipelines
2. Load all submissions and align rows by ID
3. Compute row-wise `mode()` across predictions
4. If tie (even number of models), `mode()[0]` takes the lowest class — add a tiebreaker model to avoid this

## Key Decisions

- **Odd number of models**: use 3, 5, or 7 pipelines to avoid ties
- **When to use**: discrete outputs only; for continuous predictions use rank averaging instead
- **Diversity matters**: majority voting gains nothing from correlated models — vary architecture, features, or seeds
- **Weighted variant**: `pd.Series([preds]).value_counts()` with weights for unequal-quality models

## References

- [LB0.494 with TabNet](https://www.kaggle.com/code/ichigoe/lb0-494-with-tabnet)
- [CMI | Tuning | Ensemble of solutions](https://www.kaggle.com/code/batprem/cmi-tuning-ensemble-of-solutions)
