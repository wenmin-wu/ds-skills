---
name: cv-multilabel-rare-class-image-oversampling
description: Oversample multi-label images by giving each image a duplication multiplier equal to the max per-class multiplier among its labels, so every rare class gets repetition without exploding common-class counts — the standard fix for long-tail multi-label distributions where SMOTE / per-row oversampling doesn't apply
---

## Overview

Multi-label image data has a long-tail label distribution but each image has multiple labels at once, so per-row oversampling is ambiguous: which class drives the multiplier? The HPA-winning trick is to define a per-class multiplier vector (e.g. 1 for common classes, 2-4 for rare), and for each image take the **max** multiplier across its labels. Common-only images stay at 1 copy; an image carrying any rare class gets multiplied; images with multiple rare classes still get only one bump (you don't double-multiply). This shifts the rare-class effective frequency upward without bloating common-class samples or distorting label correlations the way per-class image generation would.

## Quick Start

```python
import pandas as pd

class Oversampling:
    def __init__(self, csv_path, multi):
        df = pd.read_csv(csv_path).set_index('Id')
        df['Target'] = [[int(i) for i in s.split()] for s in df['Target']]
        self.labels = df
        self.multi = multi  # per-class duplication factor

    def get(self, image_id):
        labels = self.labels.loc[image_id, 'Target']
        return max((self.multi[l] for l in labels), default=1)

# 28-class HPA example: rare classes get 4x, common stay at 1x
multi = [1,1,1,1,1,1,1,1, 4,4,4,1,1,1,1,4,
         1,1,1,1,2,1,1,1, 1,1,1,4]
sampler = Oversampling(LABELS_CSV, multi)
train_ids = [iid for iid in train_ids for _ in range(sampler.get(iid))]
```

## Workflow

1. Compute per-class positive counts in the training set
2. Set `multi[c]` inversely proportional to count, but cap at ~4-8 (more inflates training time without lift)
3. For each image, lookup its labels and take the max multiplier
4. Materialize the oversampled ID list once at epoch start (or use a `WeightedRandomSampler` with the same ratios)
5. Shuffle the expanded list every epoch
6. Validate that the rare-class effective frequency moved closer to the head — but never to parity (full balance overfits the rare classes hard)

## Key Decisions

- **Max not sum across labels**: summing double-counts images with multiple rare classes and inflates them disproportionately.
- **Cap the multiplier at ~4-8**: 10x oversampling causes per-image overfitting (model memorizes rare-class images).
- **Hand-tune the multiplier vector**: pure inverse-frequency over-corrects; eyeball the vector and check that the effective per-class loss is close to flat.
- **Per-image, not per-pixel-augmentation**: this is sample replication, not heavier augmentation; combine the two for best results.
- **Don't apply at validation time**: oversampled validation gives misleading metrics — keep validation at natural distribution.
- **Equivalent via `WeightedRandomSampler`**: PyTorch users can pass the per-image multipliers as weights instead of materializing duplicates.

## References

- [Pretrained ResNet34 with RGBY (0.460 public LB)](https://www.kaggle.com/code/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb)
