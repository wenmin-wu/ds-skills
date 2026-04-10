---
name: tabular-logspace-recency-reranking
description: Rerank session candidates using log-spaced recency weights multiplied by interaction-type multipliers
domain: tabular
---

# Logspace Recency Reranking

## Overview

When reranking items from a user's session history, assign weights that grow logarithmically from oldest to newest event. Multiply by interaction-type multipliers (orders > carts > clicks) so recent high-intent actions dominate. Accumulate weighted scores per item via Counter and return top-K.

## Quick Start

```python
import numpy as np
from collections import Counter

def rerank_by_recency(aids, types, type_multipliers={0:1, 1:6, 2:3},
                      top_k=20):
    """Rerank candidates with logspace recency + type weights.
    
    Args:
        aids: list of item IDs in chronological order
        types: list of interaction types (0=click, 1=cart, 2=order)
        type_multipliers: weight per interaction type
        top_k: number of items to return
    """
    weights = np.logspace(0.1, 1, len(aids), base=2, endpoint=True) - 1
    scores = Counter()
    for aid, w, t in zip(aids, weights, types):
        scores[aid] += w * type_multipliers[t]
    return [aid for aid, _ in scores.most_common(top_k)]
```

## Key Decisions

- **Logspace base=2**: gentle curve; base=10 over-emphasizes the last few events
- **Subtract 1**: shifts range so oldest weight starts near 0, newest near 1
- **Type multipliers**: same as co-visitation weights for consistency across pipeline
- **Chronological order**: pass events oldest-first so logspace assigns highest weight to most recent

## References

- Source: [candidate-rerank-model-lb-0-575](https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575)
- Competition: OTTO - Multi-Objective Recommender System
