---
name: nlp-class-balanced-dataset-merge
description: >
  Merges multiple training datasets while keeping all positive examples and downsampling negatives to control class imbalance.
---
# Class-Balanced Dataset Merge

## Overview

When combining datasets from different sources or competitions, class ratios can become severely skewed. A common pattern: keep all positive (minority) examples from every source but cap the number of negatives. This gives the model maximum signal on the rare class while preventing the majority class from dominating training.

## Quick Start

```python
import pandas as pd

# Dataset 1: primary training data (use all rows)
# Dataset 2: auxiliary data (keep all positives, downsample negatives)
train = pd.concat([
    train1[["text", "label"]],
    train2[["text", "label"]].query("label == 1"),
    train2[["text", "label"]].query("label == 0").sample(n=100_000, random_state=42),
])
train = train.sample(frac=1, random_state=42).reset_index(drop=True)
```

## Workflow

1. Identify primary and auxiliary training datasets
2. Include all rows from the primary dataset
3. From each auxiliary dataset, keep all positive examples
4. Downsample auxiliary negatives to a fixed count or target ratio
5. Concatenate and shuffle before training

## Key Decisions

- **Sample count**: Set based on desired positive:negative ratio (1:5 to 1:10 is typical)
- **Random state**: Fix seed for reproducibility across experiments
- **Multiple auxiliaries**: Apply the same pattern per source; adjust counts per source quality
- **Alternative**: Use sample weights instead of dropping rows — keeps all data but upweights positives
- **Validation**: Keep validation set untouched; only rebalance training data

## References

- [Jigsaw TPU: XLM-Roberta](https://www.kaggle.com/code/xhlulu/jigsaw-tpu-xlm-roberta)
