---
name: nlp-pearson-correlation-metric
description: Use Pearson correlation coefficient as evaluation metric for semantic similarity regression tasks, selecting best checkpoint by correlation rather than loss
---

# Pearson Correlation Metric

## Overview

For regression tasks where the target represents similarity or relatedness scores, Pearson correlation measures linear agreement between predictions and labels — invariant to scale and shift. This makes it better than MSE for model selection: a model with correct ranking but wrong scale scores poorly on MSE but perfectly on Pearson r. Standard metric for STS benchmarks and similarity competitions.

## Quick Start

```python
import numpy as np
from scipy import stats

def pearson_score(y_true, y_pred):
    return stats.pearsonr(y_true, y_pred)[0]

# HuggingFace Trainer integration
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.reshape(-1)
    return {'pearson': np.corrcoef(predictions, labels)[0][1]}

trainer = Trainer(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)
```

## Workflow

1. Define `compute_metrics` returning Pearson r from predictions and labels
2. Pass to HuggingFace `Trainer` or compute manually in a training loop
3. Use Pearson r for checkpoint selection: `if score > best_score: save_model()`
4. Report final CV score as mean Pearson r across folds

## Key Decisions

- **Pearson vs Spearman**: Pearson measures linear correlation; Spearman measures rank correlation. Use Pearson when the relationship is approximately linear
- **Scale invariance**: Pearson r doesn't penalize predictions that are scaled/shifted — pair with MSE loss during training to maintain calibration
- **p-value**: `scipy.stats.pearsonr` also returns p-value — useful for small validation sets
- **Numpy shortcut**: `np.corrcoef(x, y)[0][1]` is faster than scipy for large arrays

## References

- [Getting started with NLP for absolute beginners](https://www.kaggle.com/code/jhoward/getting-started-with-nlp-for-absolute-beginners)
- [PPPM / Deberta-v3-large baseline w/ W&B [train]](https://www.kaggle.com/code/yasufuminakama/pppm-deberta-v3-large-baseline-w-w-b-train)
