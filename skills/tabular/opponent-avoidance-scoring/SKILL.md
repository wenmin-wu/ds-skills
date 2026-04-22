---
name: tabular-opponent-avoidance-scoring
description: Score candidate movement directions by average distance to nearby opponents and pick the safest path for ball-carrying agents in game AI
---

# Opponent Avoidance Scoring

## Overview

When an agent controls the ball in a game AI, choosing which direction to advance requires evaluating opponent density ahead. Score each candidate direction (right, top-right, bottom-right) by the average distance to opponents in that sector. Pick the direction with the greatest average distance — or trigger a long pass if all directions are blocked (too many opponents nearby).

## Quick Start

```python
import numpy as np

def avg_distance_to_opponents(obs, target_x, target_y):
    opponents = obs["right_team"]
    dists = [np.sqrt((ox - target_x)**2 + (oy - target_y)**2) for ox, oy in opponents]
    nearby = [d for d in dists if d < 0.05]
    return np.mean(nearby) if nearby else 2.0, len(nearby)

def pick_direction(obs, player_x, player_y):
    directions = {
        "right": (player_x + 0.01, player_y),
        "top_right": (player_x + 0.01, player_y - 0.01),
        "bottom_right": (player_x + 0.01, player_y + 0.01),
    }
    best_dir, best_dist = "right", 0
    total_nearby = 0
    for name, (tx, ty) in directions.items():
        dist, count = avg_distance_to_opponents(obs, tx, ty)
        total_nearby += count
        if dist > best_dist:
            best_dist = dist
            best_dir = name
    if total_nearby >= 3:
        return "high_pass"  # surrounded — clear the ball
    return best_dir
```

## Workflow

1. Define candidate directions as small offsets from current position
2. For each direction, compute average distance to opponents within a proximity radius
3. Pick the direction with the greatest average distance (safest path)
4. If total nearby opponents across all directions exceeds a threshold, trigger a pass/clearance
5. Add field boundary constraints (avoid going out of bounds near sidelines)

## Key Decisions

- **Proximity radius**: 0.03-0.05 in normalized coordinates; too large makes everything look blocked
- **Surrounded threshold**: 3+ nearby opponents triggers a clearance; tune based on formation width
- **Direction granularity**: 3 forward directions is sufficient for most games; add lateral for dribbling
- **Scoring function**: average distance is simple; weighted by angle to goal favors goal-ward movement
- **vs. potential fields**: scoring is cheaper to compute and easier to tune than full potential field navigation

## References

- [GFootball with Memory Patterns](https://www.kaggle.com/code/yegorbiryukov/gfootball-with-memory-patterns)
