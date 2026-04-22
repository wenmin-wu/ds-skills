---
name: tabular-hierarchical-collision-cascade
description: Three-level polygon overlap test — AABB early exit, then point-in-polygon ray casting, then segment intersection — for fast non-convex collision detection
---

# Hierarchical Collision Cascade

## Overview

Full polygon intersection is expensive for complex shapes. A three-level cascade tests cheapest first: (1) axis-aligned bounding box overlap — rejects most non-colliding pairs instantly, (2) point-in-polygon via ray casting — catches containment cases, (3) edge-edge segment intersection — catches crossing edges. Each level filters candidates for the next, achieving near-O(1) for well-separated objects.

## Quick Start

```python
import numpy as np

def aabb_overlap(a_bounds, b_bounds):
    ax0, ay0, ax1, ay1 = a_bounds
    bx0, by0, bx1, by1 = b_bounds
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)

def point_in_polygon(px, py, polygon):
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and \
           (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def segments_intersect(a1, a2, b1, b2):
    d1 = cross2d(b2 - b1, a1 - b1)
    d2 = cross2d(b2 - b1, a2 - b1)
    d3 = cross2d(a2 - a1, b1 - a1)
    d4 = cross2d(a2 - a1, b2 - a1)
    return (d1 * d2 < 0) and (d3 * d4 < 0)

def polygons_overlap(poly_a, poly_b):
    if not aabb_overlap(bounds(poly_a), bounds(poly_b)):
        return False
    for pt in poly_a:
        if point_in_polygon(pt[0], pt[1], poly_b):
            return True
    for pt in poly_b:
        if point_in_polygon(pt[0], pt[1], poly_a):
            return True
    for i in range(len(poly_a)):
        for j in range(len(poly_b)):
            if segments_intersect(
                poly_a[i], poly_a[(i+1) % len(poly_a)],
                poly_b[j], poly_b[(j+1) % len(poly_b)]):
                return True
    return False
```

## Workflow

1. **AABB check**: compare bounding boxes — if no overlap, return False immediately
2. **Point-in-polygon**: test if any vertex of A is inside B, or vice versa
3. **Segment intersection**: test all edge pairs for crossing
4. Return True only if any check at levels 2-3 passes

## Key Decisions

- **Order matters**: AABB is O(1), PIP is O(v), segment is O(v^2) — cheap filters first
- **vs Shapely**: this cascade in C/NumPy is 10-100x faster for hot loops
- **Touching vs overlapping**: shared vertices/edges need epsilon tolerance
- **Combine with STRtree**: use spatial index to find AABB candidates, then cascade for exact check

## References

- [Why Not](https://www.kaggle.com/code/jazivxt/why-not)
