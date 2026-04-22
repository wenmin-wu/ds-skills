---
name: cv-rotation-tta-segmentation
description: Test-time augmentation via 4 rotation angles (0/90/180/270), applying inverse rotation to each prediction before averaging
---

# Rotation TTA for Segmentation

## Overview

For segmentation tasks without strong orientation priors (microscopy, satellite, scroll fragments), rotating the input by 0°, 90°, 180°, and 270° and averaging the inverse-rotated predictions reduces directional bias. This is computationally efficient — all 4 rotations can be batched into a single forward pass by concatenating along the batch dimension.

## Quick Start

```python
import torch

def rotation_tta(model, x):
    B = x.shape[0]
    # Create 4 rotated versions, batch together
    rotated = [x] + [torch.rot90(x, k=k, dims=(-2, -1)) for k in range(1, 4)]
    batch = torch.cat(rotated, dim=0)  # (4*B, C, H, W)

    with torch.no_grad():
        preds = torch.sigmoid(model(batch))

    # Split and inverse-rotate
    preds = preds.reshape(4, B, *preds.shape[1:])
    aligned = [torch.rot90(preds[k], k=-k, dims=(-2, -1)) for k in range(4)]

    return torch.stack(aligned, dim=0).mean(0)

pred = rotation_tta(model, images.cuda())
```

## Workflow

1. Create 4 rotated copies of the input (0°, 90°, 180°, 270°)
2. Concatenate along batch dimension for a single forward pass
3. Apply sigmoid to raw logits
4. Split predictions back into 4 groups
5. Inverse-rotate each group by -k*90° to realign with original orientation
6. Average all 4 aligned predictions

## Key Decisions

- **4 rotations**: sufficient for most tasks; 8 (adding flips) doubles cost for marginal gain
- **Batched forward**: 4x batch size in one pass is faster than 4 separate passes
- **Memory**: if 4x batch doesn't fit, process in pairs
- **When useful**: isotropic data (no gravity direction); less useful for natural photos with orientation

## References

- [3D ResNet baseline [inference]](https://www.kaggle.com/code/yoyobar/3d-resnet-baseline-inference)
