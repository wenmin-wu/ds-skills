---
name: nlp-o-class-threshold-suppression
description: >
  Thresholds the O-class (non-entity) softmax probability in NER: if below threshold, overrides with the best non-O class to boost entity recall.
---
# O-Class Threshold Suppression

## Overview

In NER tasks where missing an entity is costly (PII detection, medical NER), the default argmax prediction is too conservative — the O class dominates and suppresses borderline entity predictions. This technique sets a high threshold (e.g., 0.99) on the O-class probability: if the model's confidence in "not an entity" is below this threshold, it falls back to the highest-scoring non-O class. This shifts the precision-recall tradeoff toward recall, catching entities the model was uncertain about.

## Quick Start

```python
import numpy as np

def suppress_o_class(logits, o_class_idx=12, threshold=0.99):
    """Override O predictions when model isn't confident enough.

    Args:
        logits: (batch, seq_len, num_classes) softmax probabilities
        o_class_idx: index of the O (non-entity) class
        threshold: minimum O-class probability to keep O prediction
    Returns:
        predictions: (batch, seq_len) label indices
    """
    preds_argmax = logits.argmax(-1)
    # Best non-O prediction
    non_o_logits = np.concatenate([
        logits[..., :o_class_idx],
        logits[..., o_class_idx+1:]
    ], axis=-1)
    preds_without_o = non_o_logits.argmax(-1)
    # Adjust indices for removed O class
    preds_without_o[preds_without_o >= o_class_idx] += 1

    o_probs = logits[..., o_class_idx]
    return np.where(o_probs < threshold, preds_without_o, preds_argmax)

predictions = suppress_o_class(softmax_outputs, o_class_idx=12, threshold=0.99)
```

## Workflow

1. Compute softmax over model logits
2. Extract argmax predictions and O-class probabilities
3. Compute best non-O class predictions
4. Where O-class probability < threshold, use non-O prediction
5. Tune threshold on validation F-beta score

## Key Decisions

- **Threshold**: 0.99 is aggressive (high recall); 0.95 is moderate; tune on validation F-beta
- **F5 metric**: With beta=5, recall is 25x more important than precision — justify aggressive thresholds
- **Per-class thresholds**: Optionally set different thresholds per entity class
- **Combining**: Stack with regex fallback for structured entities (email, phone)

## References

- [0.968 to ONNX 30-200% Speedup PII Inference](https://www.kaggle.com/code/lavrikovav/0-968-to-onnx-30-200-speedup-pii-inference)
