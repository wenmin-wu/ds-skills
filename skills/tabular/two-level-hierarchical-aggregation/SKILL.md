---
name: tabular-two-level-hierarchical-aggregation
description: >
  Aggregates deeply nested relational tables through two groupby levels (child → intermediate → parent) to build features from multi-hop relationships.
---
# Two-Level Hierarchical Aggregation

## Overview

In multi-table competitions (Home Credit, Amex, Elo), auxiliary data is often nested: a client has many loans, each loan has many monthly records. Simple one-level aggregation (monthly records → client) loses the loan-level structure. Two-level aggregation first summarizes at the intermediate level (monthly → loan), then re-aggregates at the parent level (loan → client). This preserves distributional information across the hierarchy — e.g., the client's average loan balance volatility, not just their overall average balance.

## Quick Start

```python
import pandas as pd
import numpy as np

def agg_numeric(df, group_var, prefix):
    agg = df.groupby(group_var).agg(['count', 'mean', 'max', 'min', 'sum'])
    columns = [f'{prefix}_{col}_{stat}' for col, stat in agg.columns]
    agg.columns = columns
    return agg

def agg_categorical(df, group_var, prefix):
    cat = pd.get_dummies(df.select_dtypes('object'))
    cat[group_var] = df[group_var]
    agg = cat.groupby(group_var).agg(['sum', 'mean'])
    columns = [f'{prefix}_{col}_{stat}' for col, stat in agg.columns]
    agg.columns = columns
    return agg

# Level 1: monthly records → loan
loan_agg = agg_numeric(monthly, group_var='LOAN_ID', prefix='monthly')

# Join back loan metadata
loan_features = loans[['LOAN_ID', 'CLIENT_ID']].merge(loan_agg, on='LOAN_ID')

# Level 2: loan → client
client_features = agg_numeric(
    loan_features.drop(columns=['LOAN_ID']),
    group_var='CLIENT_ID', prefix='loan'
)

# Merge into main table
train = train.merge(client_features, on='CLIENT_ID', how='left')
```

## Workflow

1. Identify the table hierarchy (parent → intermediate → child)
2. Aggregate child table by intermediate key (mean, sum, count, min, max)
3. Join intermediate-level aggregates with intermediate table metadata
4. Re-aggregate by parent key
5. Merge final features into the main training table

## Key Decisions

- **Aggregation functions**: mean/max/min/sum/count at each level; std adds value for volatility
- **Categorical handling**: One-hot encode then aggregate sum (count) and mean (proportion)
- **Memory**: Wide feature sets — apply collinearity pruning or feature selection afterward
- **Depth**: Can extend to 3+ levels, but returns diminish and features get noisy

## References

- [Introduction to Manual Feature Engineering](https://www.kaggle.com/code/willkoehrsen/introduction-to-manual-feature-engineering)
