---
name: tabular-popularity-fallback-recommendation
description: >
  Fills unfilled recommendation slots with globally popular recent items to handle cold-start users and short lists.
---
# Popularity Fallback Recommendation

## Overview

Personalized models often produce fewer than K recommendations for inactive or new users. Fill remaining slots with the most popular items from a recent time window. This simple fallback significantly boosts MAP@K by ensuring every user gets a full recommendation list.

## Quick Start

```python
import pandas as pd

def get_popular_items(transactions, days=7, top_k=12):
    cutoff = transactions["t_dat"].max() - pd.Timedelta(days=days)
    recent = transactions[transactions["t_dat"] >= cutoff]
    return recent["article_id"].value_counts().head(top_k).index.tolist()

def fill_recommendations(user_recs, popular, k=12):
    """Pad user's rec list with popular items up to k."""
    seen = set(user_recs)
    for item in popular:
        if len(user_recs) >= k:
            break
        if item not in seen:
            user_recs.append(item)
            seen.add(item)
    return user_recs[:k]
```

## Workflow

1. Compute global top-K popular items from recent time window
2. For each user, generate personalized recommendations
3. If list has fewer than K items, pad with popular items (skip duplicates)
4. Ensure every user has exactly K recommendations

## Key Decisions

- **Recency window**: 7 days typical; shorter for fast-changing catalogs
- **Deduplication**: Never recommend an item already in the personalized list
- **Ordering**: Popular items always go after personalized ones
- **Cold-start**: For users with zero history, the full list is popular items

## References

- H&M Personalized Fashion Recommendations (Kaggle)
- Source: [recommend-items-purchased-together-0-021](https://www.kaggle.com/code/cdeotte/recommend-items-purchased-together-0-021)
