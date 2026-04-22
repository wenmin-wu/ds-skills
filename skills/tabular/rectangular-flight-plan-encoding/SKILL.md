---
name: tabular-rectangular-flight-plan-encoding
description: Encode closed rectangular patrol routes as compact direction-distance strings for fleet pathfinding on toroidal game grids
---

# Rectangular Flight Plan Encoding

## Overview

In grid-based game AI competitions, units follow flight plans encoded as strings of direction characters and distances (e.g., "N3E3S3W"). A rectangular route visits the perimeter of a box and returns to the origin, maximizing cells visited for resource collection. The plan length is constrained by unit count (`floor(2 * ln(ships)) + 1`), linking fleet size to patrol range.

## Quick Start

```python
from kaggle_environments.envs.kore_fleets.helpers import Direction

def build_rect_plan(start_dir_idx, side_length):
    plan = ""
    for i in range(4):
        plan += Direction.from_index((start_dir_idx + i) % 4).to_char()
        if i < 3:
            plan += str(side_length)
    return plan

# "N5E5S5W" — a 5x5 rectangle starting north, returns to origin
plan = build_rect_plan(0, 5)

# Minimum ships needed: exp((len(plan) - 1) / 2) ≈ 21 for a 9x9 box
```

## Workflow

1. Choose starting direction (rotate across turns to cover different quadrants)
2. Set side length based on available ships and resource density
3. Build 4-leg plan: direction + distance for 3 legs, final direction implicit (auto-return)
4. Verify plan length fits within `floor(2 * ln(ships)) + 1` character limit
5. Launch fleet with the constructed plan string

## Key Decisions

- **Side length vs. ships**: larger boxes need more ships due to plan length constraint
- **Rotation per turn**: cycle starting direction (N→E→S→W) across launches to cover all quadrants
- **Mining rate**: `ln(ships)/20` per cell — many small fleets extract more than one large fleet
- **Truncation for conversion**: cut plan mid-route and append "C" to place a new structure at an offset position
- **Toroidal wrapping**: distances > board_size/2 wrap around; use this intentionally for far-side raids

## References

- [Kore Intro I: The Basics](https://www.kaggle.com/code/bovard/kore-intro-i-the-basics)
- [Kore Intro II: Mining Kore](https://www.kaggle.com/code/bovard/kore-intro-ii-mining-kore)
- [Kore Intro III: Expanding the Empire!](https://www.kaggle.com/code/bovard/kore-intro-iii-expanding-the-empire)
