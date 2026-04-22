---
name: cv-temporal-frame-jitter-augmentation
description: Add random temporal offset to the center frame during training to augment temporal diversity in video-based models
---

# Temporal Frame Jitter Augmentation

## Overview

Video-based models that extract a fixed temporal window around a labeled frame can overfit to the exact frame alignment. Adding a small random integer offset to the center frame index during training shifts the entire crop window, exposing the model to slightly different temporal contexts of the same event. This is the temporal equivalent of spatial translation augmentation.

## Quick Start

```python
import random

class VideoDataset:
    def __init__(self, frames, labels, window=24, jitter=6, mode='train'):
        self.frames = frames
        self.labels = labels
        self.window = window
        self.jitter = jitter
        self.mode = mode

    def __getitem__(self, idx):
        frame = self.frames[idx]

        if self.mode == 'train':
            frame = frame + random.randint(-self.jitter, self.jitter)

        start = frame - self.window
        end = frame + self.window + 1
        clip = self.load_frames(start, end)
        return clip, self.labels[idx]
```

## Workflow

1. Store the labeled center frame index for each sample
2. During training, add `random.randint(-jitter, jitter)` to the center frame
3. Extract the temporal window around the jittered center
4. During validation/test, use the exact center frame (no jitter)

## Key Decisions

- **Jitter range**: typically 10-25% of the window size (e.g., ±6 for window=24)
- **Train only**: never jitter during validation or inference
- **Boundary clamping**: clamp frame index to valid range to avoid out-of-bounds
- **vs temporal crop**: jitter shifts the window; random crop selects a subwindow — both are useful

## References

- [LB:0.671 2.5D CNN Baseline](https://www.kaggle.com/code/royalacecat/lb-0-671-2-5d-cnn-baseline-more-tta-trick)
- [[Training] NFL 2.5D CNN (LB:0.671 with TTA)](https://www.kaggle.com/code/royalacecat/training-nfl-2-5d-cnn-lb-0-671-with-tta)
