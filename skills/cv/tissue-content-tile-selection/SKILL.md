---
name: cv-tissue-content-tile-selection
description: >
  Selects the top-N most informative tiles from a gigapixel whole slide image by ranking on pixel intensity sum, keeping tiles with the most tissue content.
---
# Tissue-Content Tile Selection

## Overview

Whole slide images (WSIs) in histopathology are gigapixel-scale — too large for direct model input. Most of the image is white background. This technique splits the image into a grid of fixed-size tiles using numpy reshape+transpose (no Python loops), ranks tiles by pixel sum (darkest = most tissue on white-background H&E slides), and selects the top-N tiles. This is the standard first step in WSI classification pipelines, reducing a 50,000x50,000 image to 16-36 informative patches.

## Quick Start

```python
import numpy as np
import skimage.io

def extract_top_tiles(img, tile_size=256, n_tiles=36):
    """Extract top-N tiles by tissue content from WSI."""
    h, w, c = img.shape
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    img = np.pad(img, [[pad_h//2, pad_h-pad_h//2],
                        [pad_w//2, pad_w-pad_w//2], [0,0]],
                 constant_values=255)
    # Reshape into grid of tiles (no loops)
    tiles = img.reshape(img.shape[0]//tile_size, tile_size,
                        img.shape[1]//tile_size, tile_size, 3)
    tiles = tiles.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, 3)
    # Rank by pixel sum (lowest = darkest = most tissue)
    sums = tiles.reshape(tiles.shape[0], -1).sum(axis=1)
    idxs = np.argsort(sums)[:n_tiles]
    return tiles[idxs]

wsi = skimage.io.MultiImage('slide.tiff')[1]  # median resolution
tiles = extract_top_tiles(wsi, tile_size=256, n_tiles=36)
```

## Workflow

1. Load WSI at appropriate pyramid level (level 1 or 2)
2. Pad image so dimensions are divisible by tile size
3. Reshape into grid using numpy reshape+transpose
4. Sort tiles by pixel sum ascending (darkest first)
5. Select top-N tiles as model input

## Key Decisions

- **n_tiles**: 16-36 typical; more tiles = more context but slower training
- **tile_size**: 128 or 256 px; match to model input expectations
- **Pyramid level**: Level 1 (medium res) balances detail vs memory
- **Background check**: All-white tiles (sum ≈ tile_size^2 * 3 * 255) are automatically ranked last

## References

- [Train EfficientNet-B0 w/ 36 tiles](https://www.kaggle.com/code/haqishen/train-efficientnet-b0-w-36-tiles-256-lb0-87)
- [PANDA 16x128x128 tiles](https://www.kaggle.com/code/iafoss/panda-16x128x128-tiles)
