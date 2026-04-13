---
name: cv-rgby-4channel-fluorescence-loader
description: Load fluorescence microscopy images stored as 4 separate single-channel PNGs (red microtubules, green target protein, blue nucleus, yellow ER) into a single HxWx4 tensor, preserving the biological semantics of each channel rather than collapsing to RGB
---

## Overview

Fluorescence microscopy datasets like the Human Protein Atlas store one PNG per dye filter — `<id>_red.png`, `_green.png`, `_blue.png`, `_yellow.png` — because the channels carry independent biological signals (target protein, microtubules, nucleus, endoplasmic reticulum). Treating them as RGB throws away one channel and confuses the model with averaged colors. The right loader stacks all four into a single `[H, W, 4]` tensor and feeds it directly into a 4-channel network (see channel-extension techniques for adapting ImageNet pretrained models). The same pattern applies to satellite imagery (RGB+NIR), medical multi-modal scans, and any dataset where filter channels are stored separately.

## Quick Start

```python
import numpy as np
from skimage.io import imread

def load_rgby(basepath, image_id, size=512):
    img = np.zeros((size, size, 4), dtype=np.float32)
    img[:,:,0] = imread(f'{basepath}{image_id}_red.png')     # microtubules
    img[:,:,1] = imread(f'{basepath}{image_id}_green.png')   # target protein
    img[:,:,2] = imread(f'{basepath}{image_id}_blue.png')    # nucleus
    img[:,:,3] = imread(f'{basepath}{image_id}_yellow.png')  # endoplasmic reticulum
    return img / 255.0
```

## Workflow

1. Inventory the files in the data directory — confirm one image_id maps to N filter PNGs with consistent suffixes
2. Build a loader that reads each suffix and stacks them in a fixed channel order (document the order!)
3. Normalize per channel to [0, 1] or per-channel z-score if intensities vary across plates
4. Resize after stacking, not before, so all channels share the same interpolation
5. Pass to a network with conv1 expanded to N channels (transfer-learn the new channel from existing weights)
6. For TTA, augment all channels jointly — never independently

## Key Decisions

- **Stack as last axis for Keras / `H,W,C`; first axis for PyTorch `C,H,W`**: align with framework convention up front to avoid silent transposes.
- **Don't merge yellow into red+green for "RGB compatibility"**: that destroys the ER signal that distinguishes several protein localizations.
- **Document the channel order in the loader docstring**: feature visualizations and Grad-CAMs are unreadable if channel order drifts.
- **Same path pattern works for satellite RGB+NIR**: just rename the suffixes.
- **Cache decoded arrays to .npy if disk read becomes the bottleneck**: 4 PNG opens per sample is ~4x slower than one .jpg.
- **Never average channels for "grayscale fallback"**: the green channel is the target signal — if you need grayscale, use that channel alone.

## References

- [Protein Atlas - Exploration and Baseline](https://www.kaggle.com/code/allunia/protein-atlas-exploration-and-baseline)
