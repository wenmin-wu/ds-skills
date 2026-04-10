---
name: cv-stratified-fold-by-sequence-length
description: >
  Stratifies cross-validation folds by output sequence length to ensure balanced length distributions across train/val splits.
---
# Stratified Fold by Sequence Length

## Overview

In image-to-sequence tasks, output lengths vary widely (e.g., simple vs complex molecules). Random splits can create folds where one fold gets mostly short sequences and another mostly long ones, causing misleading validation scores. Stratify by binned sequence length so each fold has a representative length distribution.

## Quick Start

```python
from sklearn.model_selection import StratifiedKFold
import pandas as pd

# Compute target sequence length
train["seq_length"] = train["target_text"].str.len()

# Bin into discrete categories for stratification
train["length_bin"] = pd.qcut(train["seq_length"], q=10, labels=False, duplicates="drop")

# Create stratified folds
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train["fold"] = -1
for fold, (_, val_idx) in enumerate(skf.split(train, train["length_bin"])):
    train.loc[val_idx, "fold"] = fold
```

## Workflow

1. Compute output sequence length for each training sample
2. Bin lengths into quantile-based categories (10 bins is typical)
3. Use `StratifiedKFold` with the length bins as the stratification target
4. Assign fold numbers; each fold has similar length distribution
5. Train and validate per fold — validation metrics are now length-balanced

## Key Decisions

- **Number of bins**: 10 quantile bins is standard; more bins = stricter balance
- **Direct length vs bins**: `StratifiedKFold` needs discrete labels, so bin continuous lengths
- **Multi-target stratification**: If you have both length and class labels, use `iterative-stratification`
- **Alternatives**: Group by length ranges and sample proportionally

## References

- [InChI / Resnet + LSTM with attention / starter](https://www.kaggle.com/code/yasufuminakama/inchi-resnet-lstm-with-attention-starter)
