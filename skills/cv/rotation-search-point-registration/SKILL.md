---
name: cv-rotation-search-point-registration
description: Brute-force a 2D rotation angle over a coarse grid to align field-coordinate points with image-plane detections when the camera angle is unknown
---

## Overview

When you have two point sets — field-coordinate positions from a tracking system and image-plane detections from a camera — but the camera angle is unknown or varies per clip, standard homography fitting overfits with few points. A simpler, more robust approach: brute-force search a rotation grid (-30° to +30° in 3° steps), rotate the field coordinates at each angle, and pick the angle that minimizes 1D-normalized L2 distance against the sorted image x-centers. With 21 grid points and ~22 players, the search costs less than a millisecond per play and beats SVD-based fits on sparse, noisy data.

## Quick Start

```python
import numpy as np

def rotate_arr(u, deg):
    t = np.deg2rad(deg)
    R = np.array([[np.cos(t), -np.sin(t)],
                  [np.sin(t),  np.cos(t)]])
    return R @ u

def norm_arr(a):
    a = a - a.min()
    return a / (a.max() + 1e-9)

def best_rotation(tracking_df, image_x, deg_range=30, step=3):
    pts = tracking_df[['x','y']].values.T  # (2, N)
    best = (1e9, None, None)
    a2 = norm_arr(np.sort(image_x))
    for deg in range(-deg_range, deg_range + 1, step):
        rot = rotate_arr(pts, deg)
        rx = np.sort(rot[0])
        if len(rx) == len(a2):
            d = np.linalg.norm(norm_arr(rx) - a2)
        else:
            d = float('inf')  # handle with combinatorial deletion elsewhere
        if d < best[0]:
            best = (d, deg, rx)
    return best  # (distance, angle_deg, rotated_x)
```

## Workflow

1. Normalize both point sets to `[0, 1]` via min-max — this removes scale and translation
2. Sort along the 1D projection axis (x for side view, y for end-zone view)
3. Loop over the rotation grid and rotate one set at each angle
4. Compute L2 distance between normalized sorted arrays
5. Pick the argmin; use its angle to carry player IDs back onto the detections via sort order

## Key Decisions

- **Coarse grid first**: 3° steps are plenty; the cost function is smooth, so finer search yields diminishing returns.
- **1D projection over 2D distance**: sorting the x-axis and comparing 1D arrays is O(N) and avoids a Hungarian match for the common case.
- **vs. homography fitting**: with <10 points, SVD is unstable; grid search is deterministic and noise-robust.
- **Handle mismatched lengths separately**: use combinatorial deletion or drop-low-confidence detections first.

## References

- [Tuning DeepSort + Helmet Mapping](https://www.kaggle.com/code/its7171/tuning-deepsort-helmet-mapping)
- [NFL Baseline - Simple Helmet Mapping](https://www.kaggle.com/code/its7171/nfl-baseline-simple-helmet-mapping)
