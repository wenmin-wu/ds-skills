---
name: cv-confidence-threshold-multilabel
description: >
  Assigns multiple labels per sample using a confidence threshold on sigmoid outputs with a fallback negative class.
---
# Confidence Threshold Multi-Label Assignment

## Overview

For multi-label classification, apply a confidence threshold to sigmoid outputs to assign zero or more labels per sample. Samples with no label above the threshold receive a fallback "Negative" class with confidence `1 - max_pred`. Produces both label assignments and confidence scores per prediction.

## Quick Start

```python
import numpy as np

def multilabel_assign(probs, threshold=0.5, negative_class_id=18):
    """Assign labels from sigmoid probabilities with fallback.

    Args:
        probs: array of shape (n_classes,) with sigmoid outputs
        threshold: confidence cutoff for positive assignment
        negative_class_id: class ID for the fallback negative label

    Returns:
        list of (label_id, confidence) tuples
    """
    assignments = []
    for cls_id, p in enumerate(probs):
        if p >= threshold:
            assignments.append((cls_id, float(p)))

    if not assignments:
        assignments.append((negative_class_id, float(1.0 - probs.max())))

    return assignments


def format_submission(cell_id, assignments, mask_rle):
    """Format as competition submission string."""
    parts = []
    for label_id, conf in assignments:
        parts.append(f"{label_id} {conf:.4f} {mask_rle}")
    return " ".join(parts)
```

## Workflow

1. Run model inference to get per-class sigmoid probabilities
2. Apply threshold to select positive labels
3. If no labels pass threshold, assign negative/background class
4. Attach confidence scores for downstream calibration
5. Format predictions with associated masks/regions

## Key Decisions

- **Threshold tuning**: Optimize on validation set per-class or globally; 0.5 is a starting point
- **Per-class thresholds**: Rare classes may need lower thresholds to improve recall
- **Negative fallback**: `1 - max_pred` gives meaningful confidence for negative predictions
- **Calibration**: Sigmoid outputs are not calibrated; consider Platt scaling if scores matter

## References

- HPA Single Cell Classification competition (Kaggle)
- Source: [hpa-cellwise-classification-inference](https://www.kaggle.com/code/dschettler8845/hpa-cellwise-classification-inference)
