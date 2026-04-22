---
name: llm-spiral-patrol-exploration
description: Agents patrol in expanding spiral patterns using rotating direction sequences with increasing radius for systematic grid exploration
---

# Spiral Patrol Exploration

## Overview

In grid-based games, random movement wastes turns revisiting cells. A spiral patrol assigns each agent a rotating direction sequence (N, E, S, W) with an increasing step count per direction. The agent moves N steps north, then N steps east, then N steps south, then N steps west, then increases N. This produces an outward spiral that systematically covers the map from the agent's starting position.

## Quick Start

```python
def create_spiral_agent():
    return {
        'direction_index': 0,
        'moves_done': 0,
        'max_moves': 3,
        'directions': ['NORTH', 'EAST', 'SOUTH', 'WEST'],
    }

def spiral_step(agent):
    """Return next direction in the spiral pattern."""
    direction = agent['directions'][agent['direction_index']]
    agent['moves_done'] += 1

    if agent['moves_done'] >= agent['max_moves']:
        agent['moves_done'] = 0
        agent['direction_index'] += 1

        if agent['direction_index'] >= len(agent['directions']):
            agent['direction_index'] = 0
            agent['max_moves'] += 1
            if agent['max_moves'] > 10:
                agent['max_moves'] = 3  # reset to inner spiral

    return direction

agent = create_spiral_agent()
for _ in range(20):
    move = spiral_step(agent)
```

## Workflow

1. Initialize agent with direction sequence and starting step count
2. Each turn, move in the current direction
3. After completing `max_moves` steps, rotate to next direction
4. After completing all 4 directions, increase `max_moves` (expand spiral)
5. Reset to inner spiral when max radius is reached

## Key Decisions

- **Initial radius**: 3 steps covers nearby area before expanding
- **Max radius cap**: reset prevents agents from wandering too far from base
- **Direction order**: NESW is default; vary per agent for swarm coverage
- **Interrupts**: break spiral for high-value cells or threats, resume after

## References

- [Halite Swarm Intelligence](https://www.kaggle.com/code/yegorbiryukov/halite-swarm-intelligence)
