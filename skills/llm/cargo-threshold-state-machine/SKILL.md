---
name: llm-cargo-threshold-state-machine
description: Per-agent COLLECT/DEPOSIT state machine driven by cargo thresholds with greedy neighbor selection for resource collection games
---

# Cargo Threshold State Machine

## Overview

In resource collection games and multi-agent simulations, each agent needs a persistent decision mode. A simple two-state machine — COLLECT (gather resources) and DEPOSIT (return to base) — switches based on cargo thresholds. In COLLECT mode, the agent greedily moves toward the richest adjacent cell. In DEPOSIT mode, it navigates toward the nearest base. This baseline outperforms random movement and serves as a foundation for more complex strategies.

## Quick Start

```python
ship_states = {}

def agent_step(ship, shipyards, get_dir_fn):
    if ship.id not in ship_states:
        ship_states[ship.id] = "COLLECT"

    if ship.halite < 200:
        ship_states[ship.id] = "COLLECT"
    if ship.halite > 500:
        ship_states[ship.id] = "DEPOSIT"

    if ship_states[ship.id] == "COLLECT":
        if ship.cell.halite < 100:
            neighbors = {
                'NORTH': ship.cell.north.halite,
                'EAST': ship.cell.east.halite,
                'SOUTH': ship.cell.south.halite,
                'WEST': ship.cell.west.halite,
            }
            return max(neighbors, key=neighbors.get)
        return None  # stay and mine

    if ship_states[ship.id] == "DEPOSIT":
        return get_dir_fn(ship.position, shipyards[0].position)
```

## Workflow

1. Initialize each new agent in COLLECT state
2. Check cargo against low/high thresholds to trigger state transitions
3. COLLECT: if current cell is poor, move to richest neighbor; otherwise stay and mine
4. DEPOSIT: navigate toward nearest base using shortest-path direction
5. Persist state across turns in a global dict keyed by agent ID

## Key Decisions

- **Hysteresis**: use different thresholds for enter/exit (200/500) to prevent oscillation
- **Local vs global greedy**: 1-step neighbor check is fast; N-step lookahead finds better paths
- **Threshold tuning**: depends on map resource density — lower thresholds for sparse maps
- **Extensions**: add ATTACK, FLEE, CONVERT states for richer behavior

## References

- [Getting Started With Halite](https://www.kaggle.com/code/alexisbcook/getting-started-with-halite)
- [Designing game AI with Reinforcement learning](https://www.kaggle.com/code/basu369victor/designing-game-ai-with-reinforcement-learning)
