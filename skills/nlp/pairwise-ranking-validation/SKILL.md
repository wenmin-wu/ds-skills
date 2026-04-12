---
name: nlp-pairwise-ranking-validation
description: >
  Evaluates ranking models by computing the fraction of preference pairs where the model correctly scores the preferred item higher.
---
# Pairwise Ranking Validation

## Overview

When the task is ranking (not classification), standard metrics like accuracy or AUC don't apply directly. Pairwise ranking accuracy measures the fraction of (item_a, item_b) pairs where the model correctly assigns a higher score to the preferred item. For toxicity ranking: given pairs of (less_toxic, more_toxic) texts, check if `score(more_toxic) > score(less_toxic)`. This metric directly reflects competition evaluation and is the right validation signal for margin ranking or regression-based ranking models.

## Quick Start

```python
import numpy as np

def pairwise_accuracy(model, vectorizer, val_df):
    """Fraction of pairs correctly ordered by model scores."""
    X_less = vectorizer.transform(val_df['less_toxic'])
    X_more = vectorizer.transform(val_df['more_toxic'])

    score_less = model.predict(X_less)
    score_more = model.predict(X_more)

    return (score_less < score_more).mean()

# For transformer models
def pairwise_accuracy_nn(model, val_loader):
    scores_less, scores_more = [], []
    for batch in val_loader:
        with torch.no_grad():
            s_less = model(batch['less_ids'], batch['less_mask'])
            s_more = model(batch['more_ids'], batch['more_mask'])
        scores_less.append(s_less.cpu())
        scores_more.append(s_more.cpu())
    less = torch.cat(scores_less)
    more = torch.cat(scores_more)
    return (less < more).float().mean().item()

acc = pairwise_accuracy(ridge_model, tfidf, val_df)
print(f"Pairwise accuracy: {acc:.4f}")
```

## Workflow

1. Score both items in each validation pair independently
2. Compare: does the preferred item get a higher score?
3. Average across all pairs to get pairwise accuracy
4. Use to compare models, tune hyperparameters, select thresholds

## Key Decisions

- **Ties**: Ties count as incorrect by default; can count as 0.5 with `(less < more).mean() + 0.5 * (less == more).mean()`
- **vs Kendall's tau**: Pairwise accuracy on given pairs is simpler; Kendall's tau considers all possible pairs
- **Baseline**: Random ranking gives 0.50; a good model should be 0.70+
- **Per-fold**: Compute per CV fold and average for stable estimates

## References

- [Jigsaw ensemble best public sub](https://www.kaggle.com/code/thomasdubail/jigsaw-ensemble-best-public-sub-0-898)
- [Jigsaw Incredibly Simple Naive Bayes](https://www.kaggle.com/code/julian3833/jigsaw-incredibly-simple-naive-bayes-0-768)
