---
name: nlp-transformer-lgbm-stacking
description: >
  Two-stage stacking: transformer predictions plus hand-crafted features feed into a LightGBM meta-learner.
---
# Transformer + LightGBM Stacking

## Overview

Fine-tune a transformer (e.g., DeBERTa) for initial predictions, then use those predictions alongside hand-crafted features (text statistics, overlap metrics) as inputs to a LightGBM meta-learner. This captures both deep semantic understanding and surface-level patterns that transformers miss.

## Quick Start

```python
import lightgbm as lgb

# Stage 1: Get transformer OOF predictions per fold
for fold in range(n_folds):
    model = train_transformer(train[train.fold != fold])
    train.loc[train.fold == fold, "pred"] = predict(model, train[train.fold == fold])

# Stage 2: Stack with hand-crafted features
features = ["pred", "word_count", "overlap_ratio", "ner_overlap", "misspelling_count"]
for fold in range(n_folds):
    dtrain = lgb.Dataset(train[train.fold != fold][features], label=y_train)
    dval = lgb.Dataset(train[train.fold == fold][features], label=y_val)
    model = lgb.train(params, dtrain, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(30)])
```

## Workflow

1. Fine-tune transformer with K-fold CV, save OOF predictions
2. Engineer hand-crafted features (text stats, overlaps, spelling)
3. Combine transformer predictions + features as LightGBM input
4. Train LightGBM per fold with early stopping
5. Average fold predictions for final output

## Key Decisions

- **Why stack**: Transformers miss surface features; GBDT captures them well
- **OOF predictions**: Use out-of-fold predictions to avoid leakage in stacking
- **Early stopping**: 30 rounds patience on validation RMSE prevents overfitting
- **Per-target models**: Train separate LightGBM for each target variable

## References

- CommonLit - Evaluate Student Summaries (Kaggle)
- Source: [tuned-debertav3-lgbm-autocorrect](https://www.kaggle.com/code/cody11null/tuned-debertav3-lgbm-autocorrect)
