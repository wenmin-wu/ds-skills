---
name: cv-bigru-slice-feature-aggregator
description: Two-stage CT classifier where a 2D CNN dumps per-slice features once, then a bidirectional GRU runs over the slice sequence to produce both per-slice predictions (TimeDistributed head) and an exam-level prediction (avg+max pooled head) — turns expensive 3D CNN training into cheap sequence modeling
---

## Overview

Volumetric CT classification with a true 3D CNN is expensive: every epoch you re-encode the same slices that haven't changed. The cheaper, often-better alternative is two-stage. Stage 1 trains a 2D CNN on slices, then dumps a fixed-size feature vector per slice once and freezes. Stage 2 trains a tiny bidirectional GRU over the per-slice feature sequence, with two heads: a TimeDistributed Linear that produces per-slice predictions and a `cat(avg_pool, max_pool)` Linear that produces the exam-level prediction. Adding the inter-slice Z-gap as an extra input feature gives the GRU spatial context. Stage 2 is so cheap you can sweep dozens of hyperparameters in the time stage 1 takes for one epoch.

## Quick Start

```python
import torch
import torch.nn as nn

class TimeDistributed(nn.Module):
    def __init__(self, layer): super().__init__(); self.layer = layer
    def forward(self, x):  # (B, T, F) -> (B, T, F_out)
        B, T, F = x.shape
        return self.layer(x.reshape(B * T, F)).reshape(B, T, -1)

class SliceGRU(nn.Module):
    def __init__(self, n_feats, hidden=64, n_exam_targets=9):
        super().__init__()
        self.gru = nn.GRU(
            n_feats + 1,         # +1 for inter-slice z-gap
            hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.image_head = TimeDistributed(nn.Linear(hidden * 2, 1))
        self.exam_head  = nn.Linear(hidden * 2 * 2, n_exam_targets)

    def forward(self, slice_feats, z_gaps):
        x = torch.cat([slice_feats, z_gaps.unsqueeze(-1)], dim=2)
        h, _ = self.gru(x)                      # (B, T, 2H)
        per_slice = self.image_head(h)
        avg = h.mean(dim=1)
        mx, _ = h.max(dim=1)
        per_exam = self.exam_head(torch.cat([avg, mx], dim=1))
        return per_slice, per_exam
```

## Workflow

1. Train a 2D CNN end-to-end on per-slice classification (or load a pretrained backbone)
2. For every CT series, run the 2D CNN once and dump `(num_slices, n_feats)` to disk as a single `.npy`
3. Compute the inter-slice Z gap from `ImagePositionPatient[2]` deltas; first slice gets gap = 0
4. Train the GRU with both losses summed: per-slice BCE (with masking for padding) + per-exam BCE
5. Use `cat(mean_pool, max_pool)` for the exam-level head — single-pool is consistently worse
6. Keep the GRU tiny (hidden=64, 2 layers) — it's a sequence aggregator, not a feature extractor

## Key Decisions

- **Freeze stage 1 before dumping**: any backbone update invalidates the feature cache; freeze + dump + train stage 2 is the right order.
- **Bidirectional, not unidirectional**: PE / lesion / nodule context is symmetric; left-right context matters as much as right-left.
- **avg+max concat for exam head**: max captures "worst slice", avg captures "overall burden"; they're complementary.
- **z-gap as input feature**: lets the GRU compensate for variable slice spacing across studies.
- **Train both heads jointly**: per-slice loss provides dense supervision the per-exam head couldn't learn alone.

## References

- [CNN-GRU Baseline - Stage 2 Train+Inference](https://www.kaggle.com/competitions/rsna-str-pulmonary-embolism-detection)
