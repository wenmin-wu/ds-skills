---
name: tabular-multi-source-candidate-fusion
description: Fuse recommendation candidates from user history, multiple co-visitation matrices, and global popularity in a priority-ordered cascade
domain: tabular
---

# Multi-Source Candidate Fusion

## Overview

For recommendation tasks requiring top-K predictions, fuse candidates from multiple signal sources in priority order: (1) user's own history (recency-deduped), (2) co-visitation expansions from multiple matrices (clicks, carts/orders, buy2buy), (3) global popularity fallback. Each source fills remaining slots until K is reached. Ensures every user gets exactly K predictions regardless of history length.

## Quick Start

```python
import itertools
from collections import Counter

def fuse_candidates(session_aids, session_types, covisit_clicks,
                    covisit_buys, covisit_buy2buy, top_popular, k=20):
    """Fuse candidates from multiple sources with priority fallback.
    
    Args:
        session_aids: user's session item IDs (chronological)
        session_types: interaction types per event
        covisit_*: dict mapping aid -> list of top co-visited aids
        top_popular: list of globally popular item IDs
        k: number of candidates to return
    """
    # Priority 1: user history (recent first, deduplicated)
    unique_aids = list(dict.fromkeys(session_aids[::-1]))
    if len(unique_aids) >= k:
        return unique_aids[:k]
    
    # Priority 2: co-visitation expansion
    buy_aids = [a for a, t in zip(session_aids, session_types) if t in [1,2]]
    unique_buys = list(dict.fromkeys(buy_aids[::-1]))
    
    expanded = list(itertools.chain(
        *[covisit_buys.get(a, []) for a in unique_aids],
        *[covisit_buy2buy.get(a, []) for a in unique_buys]
    ))
    top_expanded = [a for a, _ in Counter(expanded).most_common(k)
                    if a not in set(unique_aids)]
    result = unique_aids + top_expanded[:k - len(unique_aids)]
    
    # Priority 3: global popularity fallback
    if len(result) < k:
        result += [a for a in top_popular if a not in set(result)][:k - len(result)]
    return result[:k]
```

## Key Decisions

- **Priority order**: history > co-visitation > popularity ensures personalization first
- **Multiple co-visitation sources**: click-based and buy-based matrices capture different signals
- **Counter aggregation**: items appearing in multiple co-visitation expansions rank higher
- **Reverse dedup**: `dict.fromkeys(aids[::-1])` keeps most recent occurrence of each item

## References

- Source: [candidate-rerank-model-lb-0-575](https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575)
- Competition: OTTO - Multi-Objective Recommender System
