---
name: nlp-source-balanced-stratified-fold
description: >
  Stratifies CV folds by both target label AND data source to prevent source-specific bias in each fold.
---
# Source-Balanced Stratified Fold

## Overview

When training data comes from multiple sources (different datasets, different LLMs, different collection methods), standard stratified K-fold only balances labels — individual folds may over-represent one source. Create a composite stratification key from label + source to ensure every fold has a representative mix of both dimensions.

## Quick Start

```python
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def source_balanced_split(df, label_col, source_col, n_splits=5, seed=42):
    """Stratified K-fold balanced by both label and source."""
    # Create composite key
    df['stratify_key'] = df[label_col].astype(str) + '_' + df[source_col].astype(str)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    df['fold'] = -1

    for fold, (_, val_idx) in enumerate(skf.split(df, df['stratify_key'])):
        df.loc[val_idx, 'fold'] = fold

    df.drop('stratify_key', axis=1, inplace=True)
    return df

# Usage
df = source_balanced_split(df, label_col='label', source_col='source')

# Verify balance
for fold in range(5):
    fold_df = df[df['fold'] == fold]
    print(f"Fold {fold}: {fold_df['source'].value_counts(normalize=True).to_dict()}")
```

## Workflow

1. Identify the source/origin column (dataset name, generator model, collection batch)
2. Create composite key: `f"{label}_{source}"`
3. Use StratifiedKFold with composite key
4. Verify each fold has balanced source distribution
5. Use fold assignments for CV training

## Key Decisions

- **Why not GroupKFold**: GroupKFold ensures no group leakage; this ensures balanced representation — different goals
- **Rare combinations**: If some label+source combos have <n_splits samples, merge rare sources
- **Multiple dimensions**: Extend to `label_source_prompt` for 3-way balancing if needed
- **When critical**: When sources have different difficulty levels or label quality

## References

- LLM - Detect AI Generated Text (Kaggle)
- Source: [detect-fake-text-kerasnlp-tf-torch-jax-train](https://www.kaggle.com/code/awsaf49/detect-fake-text-kerasnlp-tf-torch-jax-train)
