---
name: tabular-co-purchase-item-pairing
description: >
  Recommends items frequently purchased together with a customer's recent items using pre-computed pair dictionaries.
---
# Co-Purchase Item Pairing

## Overview

For each item, pre-compute the most frequently co-purchased item (bought by the same customer in the same session or time window). At inference, map each of a customer's recent purchases to its paired item and add those as recommendations. This captures "bought together" patterns that collaborative filtering may miss.

## Quick Start

```python
import pandas as pd
from collections import Counter

def build_pair_dict(transactions, time_window_days=7):
    """Find most common co-purchase for each item."""
    df = transactions.sort_values(["customer_id", "t_dat"])
    df["next_item"] = df.groupby("customer_id")["article_id"].shift(-1)
    df["day_diff"] = df.groupby("customer_id")["t_dat"].diff(-1).dt.days.abs()
    pairs = df[df["day_diff"] <= time_window_days]

    pair_counts = pairs.groupby("article_id")["next_item"].apply(
        lambda x: Counter(x).most_common(1)[0][0]
    )
    return pair_counts.to_dict()

pairs = build_pair_dict(transactions)
# At inference: map customer's recent items to co-purchased items
recs = [pairs.get(item) for item in customer_history if item in pairs]
```

## Workflow

1. Identify co-purchase pairs within a time window per customer
2. For each item, store the most common co-purchased partner
3. At inference, map customer's recent items through the pair dict
4. Deduplicate and append to recommendation list

## Key Decisions

- **Time window**: 1-7 days captures session-like co-purchases
- **Pair direction**: Bidirectional (A→B and B→A) or directional (sequential)
- **Fallback**: Use as supplement to main model, not standalone
- **Pre-compute**: Store pairs as dict/numpy for fast lookup at inference

## References

- H&M Personalized Fashion Recommendations (Kaggle)
- Source: [recommend-items-purchased-together-0-021](https://www.kaggle.com/code/cdeotte/recommend-items-purchased-together-0-021)
