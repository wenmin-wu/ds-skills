---
name: cv-two-stage-slice-localization-then-organ-crop
description: Use a lightweight slice-classifier to find the Z-range containing an organ in a CT volume, then crop and trilinear-resample that sub-volume into a fixed shape for a heavier 3D classifier — replaces "use the whole volume" with "use only the relevant slab" at a fraction of the FLOPs
---

## Overview

Whole-body CT volumes are 80% irrelevant for any single-organ task. Feeding all 300+ slices to a 3D CNN wastes both compute and capacity on background. The fix is a two-stage pipeline: stage 1 is a tiny 2D-or-shallow-3D model that outputs per-slice "is this slice inside the organ?" probabilities; stage 2 is a real 3D classifier that only sees the cropped Z-range above threshold, resampled to a fixed `(D, H, W)` shape. This gives the heavy model a tight field of view, lets you reuse one stage-1 detector for all organs, and turns variable-length volumes into fixed-shape tensors.

## Quick Start

```python
import torch
import torch.nn.functional as F

# Stage 1: per-slice organ presence
slice_prob = slice_net(image)                       # (D,) probabilities
mask = slice_prob > 0.5
z = torch.where(mask)[0]
z0, z1 = (z.min().item(), z.max().item() + 1) if len(z) else (0, image.shape[0])

# Stage 2: crop, resample to fixed shape, classify
sub = image[z0:z1]                                  # (d, H, W)
sub = F.interpolate(sub[None, None],
                    size=(96, 256, 256),
                    mode='trilinear',
                    align_corners=False)[0, 0]
liver_logits = liver_net(sub.unsqueeze(0))
```

## Workflow

1. Train stage 1 on slice-level labels (auto-derive from segmentation masks: slice has organ ↔ any voxel of that organ in that slice)
2. At inference, run stage 1 on the full volume; threshold at 0.5 and take the bounding Z-range
3. Pad/extend the range by a few slices to absorb localization slack
4. Crop the volume to that Z-range and trilinear-interpolate to the target `(D, H, W)`
5. Feed the fixed-shape sub-volume to the heavy organ-specific classifier
6. Repeat stage 2 per organ with the same stage-1 outputs reused

## Key Decisions

- **Threshold 0.5, not argmax**: organs straddle multiple slices; thresholding gives a contiguous range, argmax picks one.
- **Pad the Z-range by ~5 slices** on each side: stage 1 is imperfect at organ edges; padding is cheap insurance.
- **Trilinear, not nearest**: nearest creates aliasing in the through-plane direction; trilinear preserves anatomy.
- **One stage-1 model, many stage-2 models**: stage 1 is cheap to train multi-organ; stage 2 is per-organ for capacity.
- **Fall back to full volume if stage 1 returns empty mask**: empty masks occur on edge cases — defaulting to `[0, D]` is safer than skipping the patient.

## References

- [LB 0.55 — 2.5D + 3D sample model](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection)
