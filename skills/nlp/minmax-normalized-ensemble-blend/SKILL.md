---
name: nlp-minmax-normalized-ensemble-blend
description: >
  Min-max normalizes each model's predictions to [0,1] before averaging, ensuring equal contribution regardless of score distribution scale.
---
# Min-Max Normalized Ensemble Blend

## Overview

Different models produce predictions on different scales — one might output 0.01-0.99, another 0.3-0.7. Simple averaging lets the wider-range model dominate. Min-max normalizing each model's predictions to [0, 1] before blending ensures each contributes proportionally to the final ensemble regardless of its raw score range.

## Quick Start

```python
import pandas as pd

def minmax_normalize(series):
    return (series - series.min()) / (series.max() - series.min())

# Load predictions from different models
pred1 = pd.read_csv("submission_xlmr.csv")
pred2 = pd.read_csv("submission_bert.csv")

# Normalize then blend
submission = pred1.copy()
submission["toxic"] = (
    minmax_normalize(pred1["toxic"]) * 0.5 +
    minmax_normalize(pred2["toxic"]) * 0.5
)
submission.to_csv("submission.csv", index=False)
```

## Workflow

1. Generate submission files from each model independently
2. Inspect score distributions (histogram) to confirm different scales
3. Apply min-max normalization to each model's prediction column
4. Blend with equal or tuned weights
5. Optionally re-calibrate the final blend if competition metric requires it

## Key Decisions

- **When to use**: When models have visibly different score ranges; skip if ranges are similar
- **Weights**: Start with equal weights; tune via validation AUC if a holdout exists
- **Rank-based alternative**: `scipy.stats.rankdata` followed by dividing by N gives a rank-based normalization that is more robust to outliers
- **Clipping**: After blending, clip to [0, 1] if the metric expects probabilities
- **N models**: Normalize each independently, then apply weighted sum

## References

- [[TPU-Inference] Super Fast XLMRoberta](https://www.kaggle.com/code/shonenkov/tpu-inference-super-fast-xlmroberta)
