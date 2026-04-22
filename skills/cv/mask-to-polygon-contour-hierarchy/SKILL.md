---
name: cv-mask-to-polygon-contour-hierarchy
description: Convert binary segmentation masks to Shapely MultiPolygons using cv2 contour hierarchy to correctly handle interior holes, with Douglas-Peucker simplification
---

# Mask to Polygon via Contour Hierarchy

## Overview

Converting binary masks to vector polygons is needed for GIS submissions, vectorized post-processing, or spatial queries. Use `cv2.findContours` with `RETR_CCOMP` to get a two-level contour hierarchy: outer contours become polygon shells, their children become holes. Apply Douglas-Peucker simplification and minimum-area filtering to clean up jagged edges.

## Quick Start

```python
import cv2
import numpy as np
from collections import defaultdict
from shapely.geometry import Polygon, MultiPolygon

def mask_to_polygons(mask, epsilon=1.0, min_area=10.0):
    mask_uint8 = ((mask > 0) * 255).astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        mask_uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    if not contours or hierarchy is None:
        return MultiPolygon()
    approx = [cv2.approxPolyDP(c, epsilon, True) for c in contours]
    children = defaultdict(list)
    child_set = set()
    for idx, (_, _, _, parent) in enumerate(hierarchy[0]):
        if parent != -1:
            child_set.add(idx)
            children[parent].append(approx[idx])
    polys = []
    for idx, cnt in enumerate(approx):
        if idx not in child_set and cv2.contourArea(cnt) >= min_area:
            shell = cnt[:, 0, :]
            holes = [c[:, 0, :] for c in children.get(idx, [])
                     if cv2.contourArea(c) >= min_area]
            p = Polygon(shell, holes)
            if not p.is_valid:
                p = p.buffer(0)
            polys.append(p)
    return MultiPolygon(polys)
```

## Workflow

1. Threshold mask to binary uint8
2. `findContours` with `RETR_CCOMP` for two-level hierarchy
3. `approxPolyDP` to simplify contour vertices
4. Map parent-child hierarchy to shell-hole polygon pairs
5. Filter by minimum area, fix invalid geometries with `.buffer(0)`

## Key Decisions

- **RETR_CCOMP vs RETR_TREE**: CCOMP gives exactly two levels (shell + holes); TREE is for nested structures
- **epsilon**: controls simplification — higher = fewer vertices, smoother boundaries
- **min_area**: removes tiny noise polygons; tune to match annotation resolution
- **.buffer(0)**: fixes self-intersections from approximation

## References

- [Full pipeline demo: poly -> pixels -> ML -> poly](https://www.kaggle.com/code/lopuhin/full-pipeline-demo-poly-pixels-ml-poly)
