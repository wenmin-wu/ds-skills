---
name: cv-bbox-interpolation-temporal-tracking
description: Interpolate missing bounding boxes across video frames using bidirectional pandas interpolation to maintain smooth tracking through occlusions
---

# Bounding Box Interpolation for Temporal Tracking

## Overview

In video-based detection, helmet/object bounding boxes are often missing in some frames due to occlusion or detector failure. Rather than dropping those frames, collect known bbox coordinates into a DataFrame indexed by frame number, insert NaN rows for missing frames, and call `interpolate(limit_direction='both')`. This produces smooth trajectories for cropping player-centered patches across a temporal window.

## Quick Start

```python
import numpy as np
import pandas as pd

def interpolate_bboxes(detections, frame_range, subsample=4):
    """Interpolate missing bboxes across a temporal window.
    detections: DataFrame with columns [frame, left, width, top, height]
    frame_range: (start_frame, end_frame) inclusive
    subsample: take every Nth frame after interpolation
    """
    det_indexed = detections.set_index('frame')[['left', 'width', 'top', 'height']]
    bboxes = []
    for f in range(frame_range[0], frame_range[1] + 1):
        if f in det_indexed.index:
            bboxes.append(det_indexed.loc[f].values)
        else:
            bboxes.append([np.nan] * 4)

    bboxes = pd.DataFrame(bboxes, columns=['left', 'width', 'top', 'height'])
    bboxes = bboxes.interpolate(limit_direction='both').values
    return bboxes[::subsample]

bboxes = interpolate_bboxes(frame_dets, (frame - 24, frame + 24), subsample=4)
```

## Workflow

1. Query detected bounding boxes within the temporal window around the target frame
2. Build a list with known coordinates or NaN for missing frames
3. Convert to DataFrame and call `interpolate(limit_direction='both')`
4. Subsample (e.g., every 4th frame) to reduce channel count
5. Use interpolated coordinates to crop fixed-size patches from each frame

## Key Decisions

- **limit_direction='both'**: fills NaNs at both ends (forward + backward), critical when early/late frames are missing
- **Subsample rate**: every 4th frame balances temporal coverage vs. channel count
- **Group by player**: average multiple detections per frame when tracking pairs or groups
- **vs optical flow**: interpolation is simpler and sufficient when bbox drift between frames is small

## References

- [NFL 2.5D CNN Baseline [Inference]](https://www.kaggle.com/code/zzy990106/nfl-2-5d-cnn-baseline-inference)
- [[Training] NFL 2.5D CNN (LB:0.671 with TTA)](https://www.kaggle.com/code/royalacecat/training-nfl-2-5d-cnn-lb-0-671-with-tta)
