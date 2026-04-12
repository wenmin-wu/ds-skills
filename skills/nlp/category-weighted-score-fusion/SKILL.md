---
name: nlp-category-weighted-score-fusion
description: >
  Converts multi-label binary flags into a continuous regression target by applying hand-tuned per-category multipliers, then averaging across categories.
---
# Category-Weighted Score Fusion

## Overview

When a dataset has multiple binary label columns (e.g., toxic, obscene, threat, insult) but the task requires a single severity score, naive averaging treats all categories equally. Category-weighted fusion applies hand-tuned multipliers to each flag before averaging — rare/severe categories (threat, identity_hate) get higher weights, common/mild ones (obscene) get lower weights. This produces a continuous pseudo-target that better reflects severity ordering for ranking tasks.

## Quick Start

```python
import pandas as pd

# Multi-label columns with binary flags
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Per-category multipliers (tune on validation)
weights = {
    'toxic': 0.32,
    'severe_toxic': 1.5,
    'obscene': 0.16,
    'threat': 1.5,
    'insult': 0.64,
    'identity_hate': 1.5,
}

for col in label_cols:
    df[col] = df[col] * weights[col]

df['score'] = df[label_cols].mean(axis=1)
```

## Workflow

1. Start with multi-label binary columns (0/1 flags)
2. Assign per-category multipliers based on severity/rarity
3. Multiply each column by its weight
4. Average across columns to produce a single continuous score
5. Use as regression target for ranking models

## Key Decisions

- **Weight selection**: Start with inverse frequency, then tune on pairwise ranking accuracy
- **Rare categories higher**: Threat and identity_hate are rare but severe — upweight them
- **vs learned weights**: Hand-tuned weights are interpretable; learned weights (via regression) may overfit
- **Normalization**: Optional min-max scaling of final score to [0, 1]

## References

- [Try better parameters better score](https://www.kaggle.com/code/yanhf16/0-9-try-better-parameters-better-score)
