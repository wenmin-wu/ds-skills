---
name: tabular-brand-name-recovery-from-title
description: Recover missing categorical values by matching words in a related text field against a known vocabulary built from the full dataset
---

# Brand Name Recovery from Title

## Overview

When a high-cardinality categorical (e.g. brand) has many missing values but a related free-text field (e.g. product title) often contains the category value, recover missing entries by matching title words against the known vocabulary of that categorical. This is a lightweight, domain-agnostic imputation that can recover a large fraction of missing values without a model.

## Quick Start

```python
import pandas as pd

full = pd.concat([train, test])
known_brands = set(full['brand_name'].dropna().unique())

def recover_brand(row):
    if pd.isna(row['brand_name']) or row['brand_name'] == 'missing':
        for word in row['name'].split():
            if word in known_brands:
                return word
    return row['brand_name']

train['brand_name'] = train.apply(recover_brand, axis=1)
test['brand_name'] = test.apply(recover_brand, axis=1)
```

## Workflow

1. Build vocabulary: collect all unique non-null values from the categorical column across train+test
2. For each row with a missing categorical, tokenize the related text field
3. Check each token against the vocabulary set (O(1) lookup)
4. Return the first match, or keep as missing if no match found
5. Apply before encoding — the recovered values are real categories

## Key Decisions

- **Match order**: first token match wins; alternatively pick the rarest brand for specificity
- **Case sensitivity**: lowercase both sides if brand casing varies
- **Multi-word brands**: split brands into n-grams if brands contain spaces
- **Transductive**: uses test set vocab — acceptable since it's the categorical's own values, not the target

## References

- [Mercari RNN + 2Ridge models](https://www.kaggle.com/code/valkling/mercari-rnn-2ridge-models-with-notes-0-42755)
