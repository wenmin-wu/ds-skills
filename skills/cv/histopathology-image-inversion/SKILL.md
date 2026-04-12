---
name: cv-histopathology-image-inversion
description: >
  Inverts whole slide image pixel values (1 - x) so white background becomes zero, enabling standard zero-padding and making tissue regions the active signal.
---
# Histopathology Image Inversion

## Overview

H&E-stained histopathology slides have a white background (255) and colored tissue. Standard CNNs and zero-padding assume background is black (0). Inverting the image (`1.0 - x` after normalizing to [0,1]) makes the background zero and tissue non-zero. This means zero-padding naturally extends the background, and the model's normalization statistics better reflect tissue content. A simple trick that improves convergence and is standard in WSI competition pipelines.

## Quick Start

```python
import numpy as np
import torch

# Inverted mean/std (computed from 1.0 - pixel_values)
MEAN = torch.tensor([1.0 - 0.9095, 1.0 - 0.8189, 1.0 - 0.8780])
STD = torch.tensor([0.3636, 0.4998, 0.4048])

def preprocess_wsi_tile(tile):
    """Invert and normalize a WSI tile."""
    x = torch.from_numpy(tile).float() / 255.0
    x = 1.0 - x  # invert: white bg → 0, tissue → non-zero
    x = x.permute(2, 0, 1)  # HWC → CHW
    x = (x - MEAN[:, None, None]) / STD[:, None, None]
    return x
```

## Workflow

1. Load tile/patch from WSI (uint8, white background)
2. Convert to float and normalize to [0, 1]
3. Invert: `x = 1.0 - x`
4. Apply channel-wise mean/std normalization (computed on inverted data)
5. Feed to CNN with standard zero-padding

## Key Decisions

- **When to apply**: Any WSI pipeline with white-background H&E slides
- **Mean/std**: Must recompute on inverted images; don't use ImageNet stats
- **Augmentation order**: Invert before augmentation; color jitter still works normally
- **Not needed if**: Using ImageNet-pretrained models without fine-tuning (keep standard normalization)

## References

- [PANDA concat tile pooling starter](https://www.kaggle.com/code/iafoss/panda-concat-tile-pooling-starter-0-79-lb)
