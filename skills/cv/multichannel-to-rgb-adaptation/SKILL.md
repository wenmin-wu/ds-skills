---
name: cv-multichannel-to-rgb-adaptation
description: >
  Composites multi-channel imagery (microscopy, satellite) into 3-channel RGB for pretrained CNN backbones.
---
# Multi-Channel to RGB Adaptation

## Overview

Pretrained ImageNet backbones expect 3-channel RGB input. When working with multi-channel data (fluorescence microscopy, satellite multispectral, medical imaging), composite or select channels into a 3-channel image. Handles bit-depth normalization (16-bit to 8-bit) and channel selection strategy.

## Quick Start

```python
import cv2
import numpy as np

def multichannel_to_rgb(channels, bit_depth=16):
    """Composite multi-channel images to 3-channel RGB.

    Args:
        channels: dict of {"red": array, "green": array, "blue": array, ...}
        bit_depth: source bit depth (8 or 16)
    """
    r = channels["red"]
    g = channels["green"]
    b = channels["blue"]

    if bit_depth == 16:
        r = (r / 256).astype(np.uint8)
        g = (g / 256).astype(np.uint8)
        b = (b / 256).astype(np.uint8)

    return np.dstack([r, g, b])


def load_hpa_rgby(image_dir, image_id):
    """Load HPA-style RGBY channels into RGB."""
    colors = ["red", "green", "blue", "yellow"]
    channels = {}
    for c in colors:
        path = f"{image_dir}/{image_id}_{c}.png"
        channels[c] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # Option: blend yellow into red+green
    return multichannel_to_rgb(channels, bit_depth=16)
```

## Workflow

1. Load each channel as a single-channel array (preserve original bit depth)
2. Normalize to 8-bit (divide by 256 for 16-bit sources)
3. Select or blend channels into 3-channel RGB
4. Feed into pretrained backbone (ResNet, EfficientNet, etc.)

## Key Decisions

- **Channel selection**: Pick 3 most informative channels, or blend extras into existing ones
- **Normalization**: Per-channel percentile clipping often outperforms linear scaling
- **Alternative**: Modify first conv layer to accept N channels (requires unfreezing + retraining)
- **Domain**: Applies to microscopy, satellite (Sentinel-2), and medical (MRI sequences)

## References

- HPA Single Cell Classification competition (Kaggle)
- Source: [mmdetection-for-segmentation-training](https://www.kaggle.com/code/its7171/mmdetection-for-segmentation-training)
