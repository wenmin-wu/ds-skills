---
name: tabular-successive-groupby-aggregates
description: Build hierarchical features for transaction panels by aggregating twice — first groupby (entity, sub-key) to get a per-(entity, sub-key) summary, then groupby (entity) on those summaries to compute mean/min/max/std across the sub-keys, capturing the *distribution* of per-customer behavior rather than a single flat mean
---

## Overview

A flat `df.groupby('card_id').sum()` collapses a customer's whole transaction history into one number and throws away how that history is *distributed* across categories, months, or merchants. Two-step (successive) groupby keeps the distribution. Step 1: group by `(card_id, sub_key)` and compute a per-bucket statistic (sum, mean, count). Step 2: group the result by `card_id` alone and compute mean/min/max/std/skew across the buckets. The final feature for a card is no longer "average spend" but "average-of-monthly-spends, std-of-monthly-spends, max-monthly-spend" — features that distinguish a steady spender from a one-off whale even when their totals match. Every winning Elo Merchant solution leaned on this pattern.

## Quick Start

```python
import pandas as pd

# Step 1 — per (card, month_lag) summary
per_lag = (
    tx.groupby(['card_id', 'month_lag'])
      .agg(sum_amt=('purchase_amount', 'sum'),
           cnt=('purchase_amount', 'size'),
           nuniq_merch=('merchant_id', 'nunique'))
      .reset_index()
)

# Step 2 — distribution of those summaries per card
feat = per_lag.groupby('card_id').agg(
    lag_sum_mean=('sum_amt', 'mean'),
    lag_sum_std=('sum_amt', 'std'),
    lag_sum_max=('sum_amt', 'max'),
    lag_cnt_mean=('cnt', 'mean'),
    lag_nuniq_merch_mean=('nuniq_merch', 'mean'),
).reset_index()
```

## Workflow

1. Pick the entity (`card_id`, `user_id`) and the sub-key (`month_lag`, `category_2`, `merchant_id`)
2. Step 1 groupby `(entity, sub_key)` and compute base aggregates (`sum`, `count`, `nunique`, `mean`)
3. Step 2 groupby `entity` and compute distribution stats (`mean`, `std`, `min`, `max`, `skew`) across the sub-key axis
4. Suffix column names so you remember the chain: `lag_sum_std`, `cat_cnt_max`
5. Repeat for orthogonal sub-keys — `month_lag`, `category_*`, `merchant_id`, `weekday`
6. Merge each feature block onto the train/test card table

## Key Decisions

- **Always include `std` in step 2**: it is the single most predictive feature — distinguishes consistent vs. spiky behavior.
- **Compute on each transaction table separately**: `historical_transactions` and `new_merchant_transactions` get their own feature block; never concatenate them before aggregating.
- **`nunique` is expensive**: use it only on the entity-level pass, not in the inner loop.
- **Avoid mean of means without weights**: if buckets have very different sizes, also keep the count so the model can re-weight.
- **Column-name discipline**: with 5 sub-keys × 5 stats × 5 base aggs you can blow past 200 columns — use a naming scheme so feature importance plots stay readable.
- **Only the entity column joins back to train**: never leak the sub-key into the training row.

## References

- [Elo World — top-voted Elo Merchant kernel](https://www.kaggle.com/competitions/elo-merchant-category-recommendation)
