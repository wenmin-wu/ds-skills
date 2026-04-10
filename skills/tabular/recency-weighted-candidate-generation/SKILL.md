---
name: tabular-recency-weighted-candidate-generation
description: >
  Generates recommendation candidates by ranking a customer's purchase history by frequency and recency within a recent window.
---
# Recency-Weighted Candidate Generation

## Overview

For recommendation systems, generate candidates from a customer's own purchase history, weighted by how recently and frequently they bought each item. Filter to a recent time window (e.g., last 7 days of activity), count purchases per item, sort by count then date, and deduplicate. This "repurchase" signal is surprisingly strong in retail.

## Quick Start

```python
import pandas as pd

def generate_candidates(transactions, customer_id, days=7):
    cust = transactions[transactions["customer_id"] == customer_id]
    cutoff = cust["t_dat"].max() - pd.Timedelta(days=days)
    recent = cust[cust["t_dat"] >= cutoff]

    counts = recent.groupby("article_id").agg(
        ct=("t_dat", "count"),
        last_date=("t_dat", "max")
    ).reset_index()
    counts = counts.sort_values(["ct", "last_date"], ascending=False)
    return counts["article_id"].tolist()[:12]
```

## Workflow

1. Filter customer's transactions to recent activity window
2. Count purchases per item within that window
3. Sort by frequency (descending), then recency (descending) as tiebreaker
4. Deduplicate and take top-K as candidate recommendations
5. Pad remaining slots with popular items (see popularity-fallback skill)

## Key Decisions

- **Window size**: 7 days for fast-moving retail; 30+ for durable goods
- **Frequency vs recency**: Frequency first captures repeat-buy intent; recency first captures trend
- **When it works**: Strong for retail with high repurchase rates (groceries, fashion basics)
- **Combine**: Use as one candidate source alongside collaborative filtering and co-purchase

## References

- H&M Personalized Fashion Recommendations (Kaggle)
- Source: [recommend-items-purchased-together-0-021](https://www.kaggle.com/code/cdeotte/recommend-items-purchased-together-0-021)
