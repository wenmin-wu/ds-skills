---
name: cv-middle-mip-std-volume-projection
description: Compress a 3D medical volume into a 3-channel 2D image by stacking the middle slice, the max-intensity projection across depth, and the per-pixel std across depth — a poor-man's volumetric encoding that lets any pretrained 3-channel 2D CNN ingest a whole series in a single forward pass
---

## Overview

When you want to use an off-the-shelf 3-channel ImageNet backbone (RGB pretrained) but your data is volumetric, the slice-as-channel trick still requires `in_chans=N`. Even simpler: collapse the volume into exactly 3 channels by mixing complementary projections — the middle slice for anatomical context, max-intensity projection (MIP) for vessel/bright-structure highlighting, and the per-pixel standard deviation for "interesting variation" along the depth axis. Stacking these three as RGB lets a vanilla `tf_efficientnetv2_s.in1k` ingest a whole series in one forward and reach competitive scores on the RSNA Aneurysm leaderboard. The combination beats any single projection because each channel surfaces a different aspect of the volume.

## Quick Start

```python
import numpy as np

def project_volume_3ch(volume):  # volume: (D, H, W) uint8 or float
    middle = volume[len(volume) // 2]
    mip    = np.max(volume, axis=0)
    std    = np.std(volume, axis=0).astype(np.float32)
    if std.max() > std.min():
        std = ((std - std.min()) / (std.max() - std.min()) * 255).astype(np.uint8)
    else:
        std = np.zeros_like(std, dtype=np.uint8)
    return np.stack([middle, mip, std], axis=-1)  # (H, W, 3)

img = project_volume_3ch(volume)
# pass into any standard 3-channel timm model with ImageNet pretrain
```

## Workflow

1. Resample/window the volume so all values are in a comparable intensity range
2. Compute the three projections (middle slice, MIP, std-across-depth)
3. Normalize the std channel to `[0, 255]` independently — its scale is much smaller than slice intensities
4. Stack as `(H, W, 3)` and feed through the standard ImageNet normalization (mean/std)
5. Train any 2D CNN as if it were a regular RGB classification problem
6. Optional: replace one channel with a min-intensity projection for darker structures (hemorrhages, calcifications)

## Key Decisions

- **Middle slice over mean**: middle preserves contrast; mean blurs everything.
- **Why MIP**: angiographic vessels are sparse and bright — averaging dilutes them, max preserves them.
- **Why std**: depth-wise variance highlights pixels where slice-to-slice change is strongest, often around lesions and edges.
- **Independent normalization of std**: its raw scale is much smaller than intensities; if you don't rescale, it becomes a near-zero channel.
- **Beats single-projection by 1-2 LB points**: confirmed in multiple RSNA notebooks; the ensemble of three projections is the win.
- **Cheap to compute**: ~ms per volume, no model needed; good fast baseline before investing in 3D or slice-as-channel.

## References

- [RSNA-IAD | EfficientNetV2 | LB](https://www.kaggle.com/code/itsuki9180/rsna-iad-efficientnetv2-lb)
