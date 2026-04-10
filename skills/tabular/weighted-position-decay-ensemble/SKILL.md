---
name: tabular-weighted-position-decay-ensemble
description: >
  Ensembles multiple ranked recommendation lists by scoring items as model_weight / position_rank, then re-ranking.
---
# Weighted Position Decay Ensemble

## Overview

When combining ranked lists from multiple recommendation models, score each item as `model_weight / (position + 1)`. Items appearing in multiple lists accumulate scores. Re-rank by total score to produce the final list. This naturally favors items ranked high by strong models while still crediting items from weaker ones.

## Quick Start

```python
def blend_recommendations(rec_lists, weights):
    """Blend multiple ranked recommendation lists.

    Args:
        rec_lists: list of lists, each a ranked recommendation
        weights: list of floats, per-model importance weight

    Returns:
        list of items sorted by blended score
    """
    scores = {}
    for recs, w in zip(rec_lists, weights):
        for rank, item in enumerate(recs):
            scores[item] = scores.get(item, 0) + w / (rank + 1)
    ranked = sorted(scores, key=scores.get, reverse=True)
    return ranked[:12]
```

## Workflow

1. Generate ranked recommendation lists from each model
2. Assign importance weight to each model (tune on validation)
3. Score each item: accumulate `weight / (rank + 1)` across models
4. Sort by total score, take top-K

## Key Decisions

- **Decay function**: `1/(rank+1)` is simple; `1/log2(rank+2)` is gentler (NDCG-like)
- **Weight tuning**: Optimize on MAP@K validation; can use Optuna
- **Multi-stage**: Blend top models first, then re-blend result with additional models
- **Deduplication**: Built-in — items accumulate scores naturally

## References

- H&M Personalized Fashion Recommendations (Kaggle)
- Source: [lb-0-0240-h-m-ensemble-magic-multi-blend](https://www.kaggle.com/code/tarique7/lb-0-0240-h-m-ensemble-magic-multi-blend)
