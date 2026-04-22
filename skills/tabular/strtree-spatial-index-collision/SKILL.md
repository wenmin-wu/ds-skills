---
name: tabular-strtree-spatial-index-collision
description: Use Shapely STRtree spatial index for O(n log n) polygon overlap detection instead of brute-force O(n^2) pairwise checks
---

# STRtree Spatial Index Collision Detection

## Overview

Checking all pairs of polygons for overlap is O(n^2). Shapely's STRtree builds an R-tree spatial index, so `query(polygon)` returns only nearby candidates whose bounding boxes intersect. This reduces collision detection to O(n log n) in practice, enabling real-time overlap validation for packing, placement, and geospatial problems with hundreds of polygons.

## Quick Start

```python
from shapely.strtree import STRtree
from shapely.geometry import Polygon

def has_any_overlap(polygons):
    if len(polygons) <= 1:
        return False
    tree = STRtree(polygons)
    for i, poly in enumerate(polygons):
        candidates = tree.query(poly)
        for idx in candidates:
            if idx == i:
                continue
            if poly.intersects(polygons[idx]) and not poly.touches(polygons[idx]):
                return True
    return False

def find_overlapping_pairs(polygons):
    tree = STRtree(polygons)
    pairs = []
    for i, poly in enumerate(polygons):
        for idx in tree.query(poly):
            if idx > i and poly.intersects(polygons[idx]) \
               and not poly.touches(polygons[idx]):
                pairs.append((i, idx))
    return pairs
```

## Workflow

1. Build `STRtree` from all polygons (one-time O(n log n) cost)
2. For each polygon, `query` returns indices of candidates with overlapping bounding boxes
3. Run exact `intersects` only on candidates (not all n polygons)
4. Distinguish `intersects` (overlap) from `touches` (shared boundary only)
5. Rebuild tree after modifying polygon positions

## Key Decisions

- **STRtree vs grid**: STRtree handles irregular shapes; grid is faster for uniform-size objects
- **touches vs intersects**: shared edges/vertices are not overlaps — always filter with `not touches`
- **Rebuild cost**: tree must be rebuilt after moves; batch moves before rebuilding
- **Precision**: scale coordinates to integers or use `Decimal` to avoid float artifacts

## References

- [Santa 2025 - Getting Started](https://www.kaggle.com/code/inversion/santa-2025-getting-started)
