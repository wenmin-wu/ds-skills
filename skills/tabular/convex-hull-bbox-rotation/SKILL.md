---
name: tabular-convex-hull-bbox-rotation
description: Minimize axis-aligned bounding box side length by finding the optimal rotation angle over convex hull vertices using bounded scalar optimization
---

# Convex Hull Bounding Box Rotation

## Overview

An axis-aligned bounding box is rarely the tightest enclosure. Rotating the point cloud before computing the AABB can significantly reduce its area or max side length. Only the convex hull vertices matter, so reduce to hull first. Then use `scipy.optimize.minimize_scalar` to find the rotation angle that minimizes the bounding box metric — much faster than brute-force angle sweeps.

## Quick Start

```python
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial import ConvexHull

def bbox_side_at_angle(angle_deg, points):
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    rot = np.array([[c, s], [-s, c]])
    rotated = points @ rot
    span = rotated.max(axis=0) - rotated.min(axis=0)
    return max(span[0], span[1])

def optimal_bbox_rotation(all_points):
    pts = np.array(all_points)
    hull_pts = pts[ConvexHull(pts).vertices]
    initial = bbox_side_at_angle(0, hull_pts)
    res = minimize_scalar(
        lambda a: bbox_side_at_angle(a, hull_pts),
        bounds=(0.001, 89.999), method='bounded')
    if initial - res.fun > 1e-10:
        return res.fun, res.x
    return initial, 0.0

best_side, best_angle = optimal_bbox_rotation(points)
```

## Workflow

1. Collect all boundary points (polygon vertices) from placed objects
2. Compute convex hull to reduce point count
3. Define objective: max side length of AABB at rotation angle θ
4. Optimize θ in [0°, 90°) using bounded scalar minimization
5. Apply the optimal rotation to all objects for the tightest enclosure

## Key Decisions

- **Hull reduction**: checking only hull vertices is exact and O(n log n) cheaper
- **Rotation matrix**: `points @ rot.T` is vectorized and fast for large point sets
- **Search range**: [0°, 90°) covers all unique orientations by symmetry
- **Metric**: max(width, height) for square-like target; area for rectangle target

## References

- [Santa-submission](https://www.kaggle.com/code/saspav/santa-submission)
