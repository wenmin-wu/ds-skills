---
name: cv-separate-pos-neg-dice-tracking
description: Track dice score separately for positive (mask-present) and negative (empty-mask) images to avoid division distortion
---

## Overview

Standard dice metric breaks on empty masks: `2|A∩B|/(|A|+|B|) = 0/0`. Most datasets hack around this by adding a small epsilon, which distorts scores for negative images. A cleaner approach: track dice separately for positive and negative cases. For negative images, dice is simply 1 if the prediction is also empty, else 0. This gives interpretable per-category scores and reveals whether the model is failing on positives or leaking false positives on negatives.

## Quick Start

```python
import torch

def dice_metric_split(probability, truth, threshold=0.5):
    """Return (all_dice, dice_neg, dice_pos) — dice split by mask presence."""
    with torch.no_grad():
        p = (probability > threshold).float().view(len(truth), -1)
        t = (truth > 0.5).float().view(len(truth), -1)
        t_sum = t.sum(-1)
        p_sum = p.sum(-1)

        neg_idx = torch.nonzero(t_sum == 0).squeeze(-1)
        pos_idx = torch.nonzero(t_sum >= 1).squeeze(-1)

        # Negative: 1 if prediction is also empty, else 0
        dice_neg = (p_sum == 0).float()
        # Positive: standard dice
        dice_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1) + 1e-7)

        dice_neg = dice_neg[neg_idx]
        dice_pos = dice_pos[pos_idx]
    return torch.cat([dice_pos, dice_neg]), dice_neg, dice_pos
```

## Workflow

1. Binarize probability map with threshold
2. Split batch by ground-truth mask sum: zero → negative, positive → positive
3. For negatives, dice = 1 iff prediction sum is also zero (perfect empty prediction)
4. For positives, compute standard dice
5. Report both aggregates separately in logs — watch them diverge to diagnose failure modes

## Key Decisions

- **Negative definition**: Ground-truth sum == 0 (no mask pixels). Different from "negative predictions".
- **Per-image vs per-batch**: Compute per-image to avoid one large positive mask dominating the score.
- **Actionable signal**: If `dice_neg` drops, model is hallucinating masks on healthy images — add a classification gate. If `dice_pos` drops, improve segmentation directly.
- **vs. epsilon smoothing**: Epsilon hides the failure mode by giving fake dice=1 to empty pairs. Split tracking exposes it.

## References

- [UNet with ResNet34 encoder (Pytorch)](https://www.kaggle.com/code/rishabhiitbhu/unet-with-resnet34-encoder-pytorch)
