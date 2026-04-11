---
name: cv-pfbeta-threshold-optimization
description: >
  Grid-searches the optimal classification threshold to maximize probabilistic F-beta score on validation predictions.
---
# pF-beta Threshold Optimization

## Overview

Probabilistic F-beta (pFbeta) extends the standard F-beta score to work with soft predictions, weighting recall more heavily when beta > 1 (e.g., beta=1 for F1, beta=2 for recall-oriented medical screening). The default threshold of 0.5 is rarely optimal — especially with severe class imbalance (1–2% positive rate). Grid-searching over [0, 1] in 0.01 steps finds the threshold that maximizes pFbeta on validation data, often improving the score by 0.02–0.10.

## Quick Start

```python
import numpy as np
import torch

def pfbeta_torch(preds, labels, beta=1.0):
    """Probabilistic F-beta score."""
    ptp = (preds * labels).sum()
    pfp = (preds * (1 - labels)).sum()
    pfn = ((1 - preds) * labels).sum()
    precision = ptp / (ptp + pfp + 1e-10)
    recall = ptp / (ptp + pfn + 1e-10)
    return ((1 + beta**2) * precision * recall /
            (beta**2 * precision + recall + 1e-10))

def optimize_threshold(probs, labels, beta=1.0, n_steps=101):
    """Find threshold maximizing pFbeta."""
    thresholds = np.linspace(0, 1, n_steps)
    scores = []
    for t in thresholds:
        preds = (torch.tensor(probs) > t).float()
        score = pfbeta_torch(preds, torch.tensor(labels), beta).item()
        scores.append(score)
    best_idx = np.argmax(scores)
    return thresholds[best_idx], scores[best_idx]

# Usage
best_thresh, best_score = optimize_threshold(val_probs, val_labels, beta=1.0)
print(f"Best threshold: {best_thresh:.2f}, pF1: {best_score:.4f}")
test_preds = (test_probs > best_thresh).astype(int)
```

## Workflow

1. Generate soft predictions (probabilities) on validation set
2. Define pFbeta metric with desired beta value
3. Grid-search thresholds from 0.0 to 1.0 in 0.01 steps
4. Select threshold with highest pFbeta score
5. Apply to test predictions for binary submission

## Key Decisions

- **Beta value**: beta=1 for balanced F1; beta=2 for recall-oriented (medical screening)
- **Grid resolution**: 101 points (0.01 steps) is sufficient; 1001 for fine-tuning
- **Overfitting**: Threshold can overfit small validation sets — use CV-averaged threshold
- **Per-fold**: Optimize per fold and average, or optimize on all OOF predictions

## References

- [fast.ai Starter Pack - Train + Inference](https://www.kaggle.com/code/radek1/fast-ai-starter-pack-train-inference)
