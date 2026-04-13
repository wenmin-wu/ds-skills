---
name: tabular-cross-dataset-user-aggregation
description: Build user-level behavioral features (avg listing duration, relisting frequency, total items) by joining auxiliary activity tables that share user_id but not item_id with train/test
---

## Overview

Many marketplace/classifieds competitions ship auxiliary tables (`train_active`, `test_active`, `periods_*`) containing millions of extra listings that do *not* overlap with train/test on `item_id` but *do* share `user_id`. Ignoring them wastes a huge signal; joining them on `item_id` yields nothing. The right pattern is a two-step aggregation: first compute per-item statistics from the periods table (days-live, times-put-up), then mean-reduce across all of a user's items to produce user-level behavioral features, finally left-join on `user_id`. On Avito Demand Prediction this pattern was worth ~0.003 RMSE — about what a whole new model block usually buys.

## Quick Start

```python
import pandas as pd

all_samples = pd.concat([train, train_active, test, test_active]).drop_duplicates(['item_id'])
all_periods = pd.concat([train_periods, test_periods])
all_periods['days_up'] = (all_periods['date_to'] - all_periods['date_from']).dt.days

gp_df = all_periods.groupby('item_id').agg(
    days_up_sum=('days_up', 'sum'),
    times_put_up=('days_up', 'count'),
).reset_index()

item_user = all_periods.drop_duplicates('item_id')[['item_id']].merge(
    gp_df, on='item_id').merge(all_samples[['item_id', 'user_id']], on='item_id', how='left')

user_feats = item_user.groupby('user_id')[['days_up_sum', 'times_put_up']].mean().reset_index()
user_feats = user_feats.merge(
    all_samples.groupby('user_id').size().reset_index(name='n_user_items'),
    on='user_id', how='outer')

train = train.merge(user_feats, on='user_id', how='left').fillna(-1)
test  = test.merge(user_feats,  on='user_id', how='left').fillna(-1)
```

## Workflow

1. Concat `train + train_active + test + test_active` and dedupe on `item_id` to build the universe of known listings
2. Compute per-item stats from the periods table (`days_up_sum`, `times_put_up`)
3. Merge item stats onto the universe to attach `user_id`
4. Group by `user_id` and mean-reduce — one row per user, columns are behavioral averages
5. Left-join the user table onto train/test on `user_id`, fill unseen users with sentinel `-1`

## Key Decisions

- **Mean vs median at user-level**: mean captures typical behavior; median is more robust for heavy-tailed users but loses signal when most users have 1-2 listings.
- **Include `n_user_items`**: a user with 500 listings is a pro seller, which is itself predictive — don't let the mean wash that away.
- **Fill missing with `-1` not 0**: GBDTs handle `-1` as its own decision path; 0 conflates "unseen user" with "user who listed for 0 days".
- **Dedupe on `item_id` at every concat**: auxiliary tables overlap and double-counting silently inflates `n_user_items`.
- **This is not leakage**: user-level stats are computed over *all* listings including the user's test items, but they aggregate behavior rather than look up labels.

## References

- [Aggregated features & LightGBM](https://www.kaggle.com/code/bminixhofer/aggregated-features-lightgbm)
