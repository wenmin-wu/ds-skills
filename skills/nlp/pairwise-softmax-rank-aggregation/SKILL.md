---
name: nlp-pairwise-softmax-rank-aggregation
description: >
  Converts pairwise binary predictions into continuous ranks via temperature-scaled softmax weighted sum over anchor positions.
---
# Pairwise Softmax Rank Aggregation

## Overview

In pairwise ranking, a model predicts P(A > B) for every pair (A, B). To recover a global ordering, apply temperature-scaled softmax over pairwise scores for each query item, then compute a weighted sum of anchor positions. The temperature controls sharpness: high temperature produces a soft average, low temperature approaches argmax. This converts N binary predictions into a single continuous rank value.

## Quick Start

```python
import numpy as np

def pairwise_to_rank(pairwise_preds, anchor_positions, temperature=20):
    """Convert pairwise scores to a continuous rank.

    Args:
        pairwise_preds: array of P(query > anchor_i) for each anchor
        anchor_positions: known positions of anchor items (e.g., code cell ranks)
        temperature: higher = sharper softmax (default 20)
    """
    centered = pairwise_preds - np.mean(pairwise_preds)
    weights = np.exp(centered * temperature)
    weights /= weights.sum()
    return np.sum(weights * anchor_positions)

# For each markdown cell, predict rank relative to all code cells
for i, md_preds in enumerate(all_pairwise_preds):
    ranks[i] = pairwise_to_rank(md_preds, code_cell_ranks)
```

## Workflow

1. Train a binary classifier: P(item_A should come before item_B)
2. At inference, predict pairwise scores for each query against all anchors
3. Center predictions (subtract mean) for numerical stability
4. Apply temperature-scaled softmax to get attention weights
5. Weighted sum of anchor positions gives the continuous rank

## Key Decisions

- **Temperature**: 10-30 typical; higher makes ranking sharper, lower gives smoother averages
- **Anchor set**: Use items with known positions (e.g., code cells with fixed relative order)
- **Centering**: Subtract mean before softmax to prevent overflow
- **Alternative**: Simple argmax of pairwise scores, but loses granularity

## References

- [AI4Code Pairwise BertSmall inference](https://www.kaggle.com/code/yuanzhezhou/ai4code-pairwise-bertsmall-inference)
