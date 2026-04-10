---
name: tabular-weighted-recall-multi-objective-metric
description: Evaluate recommendation quality with recall@K per action type, combined via business-importance weights
domain: tabular
---

# Weighted Recall Multi-Objective Metric

## Overview

For multi-objective recommendation (predict clicks, carts, and orders), evaluate recall@K separately per action type, then combine with weights reflecting business importance. Orders matter most (0.60), carts next (0.30), clicks least (0.10). This single metric aligns optimization with business value while tracking per-type performance.

## Quick Start

```python
import numpy as np

def weighted_recall(predictions, ground_truth, k=20,
                    weights={'clicks': 0.10, 'carts': 0.30, 'orders': 0.60}):
    """Compute weighted recall across action types.
    
    Args:
        predictions: dict {action_type: {session_id: [predicted_aids]}}
        ground_truth: dict {action_type: {session_id: [true_aids]}}
        k: max predictions per session
        weights: importance weight per action type
    """
    recalls = {}
    for action_type, w in weights.items():
        hits, total = 0, 0
        preds = predictions.get(action_type, {})
        truth = ground_truth.get(action_type, {})
        for sid, true_aids in truth.items():
            pred_aids = set(preds.get(sid, [])[:k])
            true_set = set(true_aids)
            hits += len(pred_aids & true_set)
            total += min(len(true_set), k)
        recalls[action_type] = hits / max(total, 1)
    
    score = sum(recalls[t] * w for t, w in weights.items())
    return score, recalls
```

## Key Decisions

- **60/30/10 split**: orders drive revenue; adjust weights to match your business priorities
- **Recall not precision**: users see K slots, so covering ground truth matters more than avoiding false positives
- **Per-type tracking**: monitor individual recalls to catch type-specific regressions
- **Cap at K**: `min(len(true_set), k)` avoids penalizing when ground truth exceeds K

## References

- Source: [otto-getting-started-eda-baseline](https://www.kaggle.com/code/edwardcrookenden/otto-getting-started-eda-baseline)
- Competition: OTTO - Multi-Objective Recommender System
