---
name: tabular-time-varying-reward-shaping
description: Shape RL rewards with time-decaying asset weights and time-increasing resource weights so the agent transitions from expansion to accumulation as the game progresses
---

# Time-Varying Reward Shaping

## Overview

In resource-management games, the optimal strategy shifts over time: early game rewards fleet/structure building (assets), late game rewards resource hoarding. Time-varying reward shaping encodes this by linearly interpolating weights between assets and resources as a function of turn number, plus a terminal win bonus inversely proportional to game length.

## Quick Start

```python
import numpy as np

max_steps = 400
transition_point = 300

w_assets = np.concatenate([
    np.linspace(1.0, 0.0, transition_point),
    np.zeros(max_steps - transition_point)
])
w_resources = np.concatenate([
    np.linspace(0.0, 1.0, transition_point),
    np.ones(max_steps - transition_point)
])

def board_value(player, turn, ship_cost=10, yard_cost=50):
    val_resources = w_resources[turn] * player.kore
    val_ships = w_assets[turn] * ship_cost * player.total_ships
    val_yards = w_assets[turn] * yard_cost * player.total_shipyards
    return val_resources + val_ships + val_yards

def reward(board, prev_board, turn, done, won):
    r = board_value(board.current_player, turn) - board_value(prev_board.current_player, turn - 1)
    if done:
        bonus = (1 if won else -1) * (100 + 5 * (max_steps - turn))
        r += bonus
    return r
```

## Workflow

1. Define linear weight schedules for assets and resources over the game horizon
2. Compute board value as weighted sum of resources, fleet value, and structure value
3. Reward = delta board value between consecutive steps
4. Add terminal bonus/penalty scaled by remaining turns (faster wins → bigger bonus)
5. Tune the transition point and weight slopes on validation episodes

## Key Decisions

- **Transition point**: set to ~75% of max steps; too early and the agent hoards prematurely
- **Asset valuation**: weight structures more than ships (shipyards compound value over time)
- **Win bonus scaling**: `remaining_turns * constant` encourages decisive play over stalling
- **Delta vs. absolute**: delta reward (change between steps) gives denser signal than absolute value

## References

- [Reinforcement Learning baseline in Python](https://www.kaggle.com/code/lesamu/reinforcement-learning-baseline-in-python)
