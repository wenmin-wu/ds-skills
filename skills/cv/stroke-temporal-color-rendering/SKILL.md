---
name: cv-stroke-temporal-color-rendering
description: Render stroke sequences to grayscale images with temporal intensity encoding where earlier strokes are brighter and later strokes fade to encode drawing order
---

# Stroke Temporal Color Rendering

## Overview

Hand-drawn sketches are stored as ordered stroke sequences. When rendering to an image, encoding the drawing order as pixel intensity — earlier strokes brighter, later strokes darker — gives CNNs an additional temporal signal beyond pure geometry. This simple trick improves doodle classification accuracy by 1-3% over flat-color rendering.

## Quick Start

```python
import cv2
import numpy as np
import json

def draw_strokes(raw_strokes, size=64, line_width=6):
    img = np.zeros((256, 256), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        color = max(255 - t * 13, 30)  # fade from 255 to ~30
        for i in range(len(stroke[0]) - 1):
            cv2.line(img,
                     (stroke[0][i], stroke[1][i]),
                     (stroke[0][i + 1], stroke[1][i + 1]),
                     color, line_width)
    if size != 256:
        img = cv2.resize(img, (size, size))
    return img

strokes = json.loads(drawing_string)
img = draw_strokes(strokes, size=64)
```

## Workflow

1. Parse stroke JSON: list of `[[x0, x1, ...], [y0, y1, ...]]` per stroke
2. Create a blank canvas (256x256 grayscale)
3. Draw each stroke with intensity `255 - t * step`, clamped to a minimum
4. Resize to target CNN input size (64x64 or 128x128)
5. Normalize to [0, 1] or apply model-specific preprocessing

## Key Decisions

- **Fade rate**: `13 per stroke` works for ~20-stroke drawings; adjust for longer sequences
- **Minimum intensity**: clamp at 30 to keep late strokes visible
- **Line width**: 6px at 256x256 base; scale proportionally for smaller canvases
- **vs. flat color**: temporal encoding adds ~1-3% top-3 accuracy at zero computational cost
- **vs. RGB channels**: encode first/middle/last strokes in R/G/B for richer signal with pretrained models

## References

- [Greyscale MobileNet [LB=0.892]](https://www.kaggle.com/code/gaborfodor/greyscale-mobilenet-lb-0-892)
