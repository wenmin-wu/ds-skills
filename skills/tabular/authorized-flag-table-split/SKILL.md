---
name: tabular-authorized-flag-table-split
description: Split a transaction table by a binary status flag (authorized vs. declined, paid vs. refunded) into two parallel sub-tables, then build the same aggregate feature pipeline on each — the declined-transaction features are usually as predictive as the authorized ones because they encode risk and friction the authorized stream alone hides
---

## Overview

A common mistake on transaction panels is to filter `authorized_flag == 'Y'` and discard the rest. The declined rows are not noise — they encode declined-rate, declined-amount, declined-merchant-diversity, and other friction signals that strongly correlate with churn, fraud, and loyalty score. Top Elo Merchant solutions all built three parallel feature blocks: (1) full table, (2) authorized-only sub-table, (3) non-authorized sub-table — each through the same `groupby(card_id).agg(...)` pipeline. The model learns from the *contrast*: `auth_amt_mean / nonauth_amt_mean`, `auth_count / total_count`. Generalize to any binary flag — paid/refunded, success/failure, mobile/web.

## Quick Start

```python
auth    = tx[tx.authorized_flag == 'Y']
nonauth = tx[tx.authorized_flag == 'N']

def agg_block(df, prefix):
    g = df.groupby('card_id').agg(
        amt_sum=('purchase_amount', 'sum'),
        amt_mean=('purchase_amount', 'mean'),
        cnt=('purchase_amount', 'size'),
        nuniq_merch=('merchant_id', 'nunique'),
    ).reset_index()
    g.columns = ['card_id'] + [f'{prefix}_{c}' for c in g.columns[1:]]
    return g

feat_all  = agg_block(tx,      'all')
feat_auth = agg_block(auth,    'auth')
feat_non  = agg_block(nonauth, 'nonauth')

cards = feat_all.merge(feat_auth, on='card_id', how='left')\
                .merge(feat_non,  on='card_id', how='left')
cards['auth_rate'] = cards['auth_cnt'] / cards['all_cnt']
```

## Workflow

1. Identify the binary flag (`authorized_flag`, `paid`, `is_success`)
2. Split the transaction frame into two sub-frames — keep both, do not filter and forget
3. Run the same aggregate pipeline (sum/mean/count/nunique/std) on each sub-frame, plus the union
4. Suffix column names (`auth_*`, `nonauth_*`) so feature importance is self-explanatory
5. Hand-craft 2-3 ratio features post-merge (`auth_rate`, `auth_amt_share`)
6. Left-merge all three blocks onto the entity table; fill NaN with 0 only when 0 is semantically zero (e.g., count), not for means

## Key Decisions

- **Three blocks not two**: the union block is not redundant — it gives the model an unconditional baseline that the splits are deviations from.
- **Ratio features are critical**: `auth_rate` (declined fraction) is usually a top-10 feature on its own.
- **Fill strategy matters**: counts → 0; means/stds → leave NaN so LightGBM can split on missingness directly.
- **Generalizes beyond yes/no**: any low-cardinality status column (3-5 values) can be split this way; cardinality > 10 is too many parallel blocks.
- **Don't double-count in the union**: the union features come from `tx`, not `pd.concat([auth, nonauth])` — they are identical but the latter is slower.

## References

- [Elo World — top-voted Elo Merchant kernel](https://www.kaggle.com/competitions/elo-merchant-category-recommendation)
