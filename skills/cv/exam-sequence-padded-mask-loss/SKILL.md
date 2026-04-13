---
name: cv-exam-sequence-padded-mask-loss
description: Pad variable-length per-slice sequences to a fixed batch length, carry a 0/1 mask alongside, and multiply per-slice BCE by the mask before reducing — gives correct per-exam loss with batched training and zero contamination from padding tokens
---

## Overview

Variable-length CT series (one study has 200 slices, another has 800) cannot be batched directly — torch needs rectangular tensors. The standard fix is to pad to `max_slices`, but the trap is that any per-slice loss now includes the padded zeros and produces a meaningless gradient. Carrying a `(B, max_slices)` 0/1 mask and multiplying it into the per-element BCE before sum-and-divide is the fix. Crucially, normalize by the *real* slice count `mask.sum()`, not by `max_slices` — otherwise short studies produce artificially small losses and the optimizer ignores them. The same pattern applies to any padded-sequence regression (NLP token loss, point-cloud labeling, time-series anomaly per-step heads).

## Quick Start

```python
import torch
import torch.nn.functional as F

def padded_per_slice_bce(y_pred, y_true, mask, label_weight=1.0):
    """
    y_pred, y_true: (B, T) per-slice logits and targets
    mask:           (B, T) 1.0 for real slices, 0.0 for padding
    """
    raw = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
    raw = raw * mask                                # zero out padding
    # per-exam normalization by real slice count
    n_real = mask.sum(dim=1).clamp(min=1.0)         # (B,)
    per_exam_loss = raw.sum(dim=1) / n_real          # (B,)
    # downweight short exams less aggressively with sqrt or log if needed
    return (label_weight * per_exam_loss).mean()
```

## Workflow

1. In `__getitem__`, pad slice features to `max_slices` with zeros and produce a mask: `mask[:n_real] = 1.0`
2. Stack into batch tensors `(B, max_slices, n_feats)` and `(B, max_slices)` for the mask
3. In the loss, compute per-element BCE with `reduction='none'` then multiply by the mask
4. Sum over the time axis and divide by `mask.sum(dim=1).clamp(min=1)` to get per-exam loss
5. Mean over the batch (or sum if combining with other heads at exam level)
6. Apply the same masking inside any metric (AUROC, accuracy) — sklearn doesn't know about padding

## Key Decisions

- **`reduction='none'` then mask**: never `reduction='mean'` followed by mask; the mean already divided by the wrong denominator.
- **Normalize by `mask.sum()`, not `max_slices`**: the latter underweights short exams by a factor of `n_real / max_slices`.
- **`clamp(min=1)` on the divisor**: defends against an empty exam in the batch (rare but possible with bad data).
- **Mask is float, not bool**: BCE multiplication needs float; bool works for simple slicing but not for the multiplication path.
- **Apply the mask to the prediction *before* sigmoid in any per-slice metric**: don't compute AUC over padded zeros, you'll get garbage.

## References

- [CNN-GRU Baseline - Stage 2 Train+Inference](https://www.kaggle.com/competitions/rsna-str-pulmonary-embolism-detection)
