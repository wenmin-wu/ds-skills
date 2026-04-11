---
name: tabular-confidence-probability-clipping
description: >
  Hard-clips predicted probabilities to 0 or 1 when they exceed high-confidence thresholds, reducing log loss on near-certain predictions.
---
# Confidence Probability Clipping

## Overview

Log loss penalizes confident wrong predictions exponentially — predicting 0.99 when the true label is 0 costs 100x more than predicting 0.5. But for predictions the model is very confident about, the risk of being wrong is low. Clipping probabilities above/below asymmetric thresholds (e.g., >0.86→1.0, <0.14→0.0) converts these near-certain soft predictions to hard ones. When correct, this eliminates residual log loss from the epsilon gap; when wrong, the loss is bounded by the threshold. On balanced log loss metrics, this typically improves score by 0.001-0.01.

## Quick Start

```python
import numpy as np

def clip_confident_predictions(probs, upper=0.86, lower=0.14):
    """Hard-clip probabilities at confidence thresholds.

    Args:
        probs: predicted P(class_0), shape (n,)
        upper: threshold above which to clip to 1.0
        lower: threshold below which to clip to 0.0
    Returns:
        Clipped probabilities
    """
    clipped = probs.copy()
    clipped[clipped > upper] = 1.0
    clipped[clipped < lower] = 0.0
    return clipped

p0 = model.predict_proba(X_test)[:, 0]
p0 = clip_confident_predictions(p0, upper=0.86, lower=0.14)
submission['class_0'] = p0
submission['class_1'] = 1 - p0
```

## Workflow

1. Get predicted probabilities from model/ensemble
2. Set upper and lower confidence thresholds
3. Clip probabilities beyond thresholds to 0 or 1
4. Ensure complementary probabilities sum to 1

## Key Decisions

- **Asymmetric thresholds**: Different thresholds per class (e.g., 0.86/0.14 or 0.60/0.26) reflect different confidence levels
- **Threshold tuning**: Grid-search on validation balanced log loss; start symmetric, then try asymmetric
- **Risk**: Wrong hard predictions are very expensive — only clip when ensemble is strong
- **After ensemble**: Apply clipping after ensembling, not to individual models

## References

- [Postprocessin_ Ensemble](https://www.kaggle.com/code/vadimkamaev/postprocessin-ensemble)
