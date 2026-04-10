---
name: cv-multilabel-auc-evaluation
description: >
  Computes per-class ROC-AUC then macro-averages for multi-label classification evaluation and model selection.
---
# Multi-Label AUC Evaluation

## Overview

For multi-label classification (multiple binary labels per sample), compute ROC-AUC independently for each label, then average. This macro-averaged AUC treats rare and common labels equally, preventing the model from ignoring minority classes. Use as the primary metric for model selection and early stopping.

## Quick Start

```python
import numpy as np
from sklearn.metrics import roc_auc_score

def multilabel_auc(y_true, y_pred, label_names=None):
    """Compute per-label and macro-averaged AUC.

    Args:
        y_true: array (n_samples, n_labels), binary ground truth
        y_pred: array (n_samples, n_labels), predicted probabilities
        label_names: optional list of label names for reporting
    """
    per_label = {}
    for i in range(y_true.shape[1]):
        # Skip labels with single class in this split
        if len(np.unique(y_true[:, i])) < 2:
            continue
        name = label_names[i] if label_names else f'label_{i}'
        per_label[name] = roc_auc_score(y_true[:, i], y_pred[:, i])

    macro_auc = np.mean(list(per_label.values()))
    return macro_auc, per_label

# Usage in validation loop
y_preds = model(images).sigmoid().cpu().numpy()
macro_auc, per_label = multilabel_auc(y_true, y_preds, target_cols)
print(f"Macro AUC: {macro_auc:.4f}")
for name, auc in per_label.items():
    print(f"  {name}: {auc:.4f}")
```

## Workflow

1. Collect all validation predictions (sigmoid probabilities, not logits)
2. Compute ROC-AUC per label independently
3. Skip labels with only one class in the validation set (AUC undefined)
4. Macro-average across all valid labels
5. Use macro AUC for early stopping and model checkpointing

## Key Decisions

- **Macro vs micro**: Macro weights all labels equally; micro favors majority labels
- **Sigmoid vs softmax**: Multi-label uses sigmoid (independent per label); multi-class uses softmax
- **Threshold-free**: AUC doesn't require choosing a threshold — better for model selection
- **Per-label tracking**: Monitor per-label AUC to catch labels where the model struggles

## References

- RANZCR CLiP - Catheter and Line Position Challenge (Kaggle)
- Source: [ranzcr-resnext50-32x4d-starter-training](https://www.kaggle.com/code/yasufuminakama/ranzcr-resnext50-32x4d-starter-training)
