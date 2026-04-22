---
name: tabular-toroidal-manhattan-distance
description: Compute shortest Manhattan distance on a toroidal (wrapping) grid by comparing normal vs wrap-around routes in each axis
---

# Toroidal Manhattan Distance

## Overview

On a toroidal grid (edges wrap around), the shortest path between two points may go through the boundary. For each axis, compare the direct distance with the wrap-around distance and take the minimum. This is essential for pathfinding, nearest-neighbor search, and feature engineering on any game or simulation with periodic boundary conditions.

## Quick Start

```python
def toroidal_manhattan(x1, y1, x2, y2, size):
    """Shortest Manhattan distance on a size x size wrapping grid."""
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    dx = min(dx, size - dx)
    dy = min(dy, size - dy)
    return dx + dy

def toroidal_direction(from_pos, to_pos, size):
    """Best direction to move toward target on toroidal grid."""
    dx = (to_pos[0] - from_pos[0]) % size
    dy = (to_pos[1] - from_pos[1]) % size
    best = None
    if dx != 0:
        best = 'EAST' if dx <= size // 2 else 'WEST'
    if dy != 0:
        d = 'NORTH' if dy <= size // 2 else 'SOUTH'
        best = d if best is None else best
    return best

dist = toroidal_manhattan(3, 18, 19, 2, size=21)  # wraps in both axes
```

## Workflow

1. Compute absolute difference in each axis
2. Compare with `size - diff` (wrap-around route)
3. Take minimum per axis, sum for Manhattan distance
4. For direction: use modular arithmetic to determine shortest wrap direction

## Key Decisions

- **Grid size**: must match the simulation's board dimensions
- **Euclidean variant**: replace sum with `sqrt(dx² + dy²)` for Euclidean distance
- **Batch computation**: vectorize with numpy for all-pairs distance matrices
- **Applications**: game AI pathfinding, spatial clustering on periodic domains, molecular simulations

## References

- [Halite Swarm Intelligence](https://www.kaggle.com/code/yegorbiryukov/halite-swarm-intelligence)
