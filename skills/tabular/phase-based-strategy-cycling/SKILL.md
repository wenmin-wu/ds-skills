---
name: tabular-phase-based-strategy-cycling
description: Divide a game into repeating phases (attack, mine, spawn) with turn-modular gating so the agent cycles between aggressive and economic behavior
---

# Phase-Based Strategy Cycling

## Overview

In multi-phase game AI, a single static strategy loses to adaptive play. Dividing the game into repeating cycles — e.g., attack (turns 0-20), mine (20-90), idle/spawn (90-100) within each 100-turn block — lets a heuristic agent alternate between aggression and economy. Each phase has different action priorities and ship allocation rules.

## Quick Start

```python
def agent(obs, config):
    board = Board(obs, config)
    me = board.current_player
    turn = board.step
    cycle_pos = turn % 100

    for shipyard in me.shipyards:
        if cycle_pos < 20 and shipyard.ship_count >= 50:
            launch_attack(shipyard, board)
        elif 20 <= cycle_pos < 90 and shipyard.ship_count >= 10 and turn % 7 == 0:
            launch_mining_fleet(shipyard, turn)
        elif me.kore >= board.configuration.spawn_cost * shipyard.max_spawn:
            shipyard.next_action = ShipyardAction.spawn_ships(shipyard.max_spawn)

    return me.next_actions
```

## Workflow

1. Define cycle length (e.g., 100 turns) and phase boundaries
2. Attack phase: accumulate ships, launch large fleets at nearest enemy structures
3. Mining phase: send periodic small fleets on patrol routes, rotate directions
4. Spawn phase: spend excess resources on ship production for the next cycle
5. Add early-game override: pure spawning for the first N turns to build up economy
6. Tune phase boundaries and thresholds via self-play evaluation

## Key Decisions

- **Cycle length**: 80-120 turns; shorter cycles are more aggressive, longer ones more economic
- **Attack threshold**: only attack when ship count exceeds enemy garrison + safety margin
- **Mining frequency**: every 5-10 turns per shipyard to avoid fleet congestion
- **Late-game override**: in final 10% of game, switch to pure resource hoarding regardless of phase
- **Adaptive cycling**: advanced agents adjust phase lengths based on opponent behavior (detect if enemy is mining-heavy → extend attack phase)

## References

- [Kore Intro IV: Combat!](https://www.kaggle.com/code/bovard/kore-intro-iv-combat)
