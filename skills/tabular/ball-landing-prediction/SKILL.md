---
name: tabular-ball-landing-prediction
description: Predict where a projectile will land using kinematic equations with estimated gravity to intercept aerial passes in game AI simulations
---

# Ball Landing Prediction

## Overview

In physics-based game AI, aerial balls follow parabolic trajectories. Instead of chasing the ball's current position, compute where it will land using kinematic equations and run to the intercept point. This requires estimating the game's gravity constant from observation logs, then solving for landing time and applying horizontal velocity.

## Quick Start

```python
import numpy as np

GRAVITY = 0.098  # estimated from environment observation logs
PICK_HEIGHT = 0.5  # height at which a player can control the ball

def predict_landing(ball_pos, ball_dir):
    z0 = ball_pos[2]
    vz = ball_dir[2]
    vx, vy = ball_dir[0], ball_dir[1]

    discriminant = (vz / GRAVITY) ** 2 - 2 * (PICK_HEIGHT - z0) / GRAVITY
    if discriminant < 0:
        return ball_pos[:2]  # ball won't reach pick height
    t = vz / GRAVITY + np.sqrt(discriminant)

    land_x = ball_pos[0] + vx * t
    land_y = ball_pos[1] + vy * t
    return [land_x, land_y]
```

## Workflow

1. Log ball position and velocity over multiple steps to estimate gravity (`delta_vz` per step)
2. When ball height > pick threshold, compute landing time via quadratic formula
3. Project horizontal position forward by landing time: `x + vx * t`
4. Move the defender/attacker toward the predicted landing point
5. If ball is below pick height, chase its current position directly

## Key Decisions

- **Gravity estimation**: run a calibration episode and log `ball_direction[2]` changes; typically constant
- **Pick height**: the height at which a player can intercept — game-specific, discover via testing
- **Air drag**: most game engines ignore drag; if present, apply a decay factor to horizontal velocity
- **Fallback**: if discriminant < 0 (ball rising), use current ball position until it starts descending
- **Application**: same technique works for thrown/kicked objects in any physics simulation

## References

- [GFootball Rules from Environment Exploration](https://www.kaggle.com/code/sx2154/gfootball-rules-from-environment-exploration)
