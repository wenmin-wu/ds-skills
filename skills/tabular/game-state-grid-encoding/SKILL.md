---
name: tabular-game-state-grid-encoding
description: Encode a 2D game board into a normalized multi-channel feature tensor with log-scaled resources, signed unit counts, and directional features for RL agents
---

# Game State Grid Encoding

## Overview

Game AI competitions often provide board state as nested objects (cells, fleets, shipyards). Converting this into a fixed-size tensor with per-cell feature channels enables CNN or MLP-based RL agents. Each channel encodes a different aspect: log-scaled resources, signed unit counts (+1 for friendly, -1 for enemy), direction values, and structure presence.

## Quick Start

```python
import numpy as np

N_FEATURES = 4  # kore, ships, direction, shipyard

def encode_board(board, player_id):
    size = board.configuration.size
    state = np.zeros((size, size, N_FEATURES), dtype=np.float32)
    for point, cell in board.cells.items():
        state[point.y, point.x, 0] = cell.kore
        if cell.fleet:
            sign = 1 if cell.fleet.player_id == player_id else -1
            state[point.y, point.x, 1] = sign * cell.fleet.ship_count
            state[point.y, point.x, 2] = cell.fleet.direction.value
        if cell.shipyard:
            state[point.y, point.x, 3] = 1 if cell.shipyard.player_id == player_id else -1
    state[:, :, 0] = np.clip(np.log2(state[:, :, 0] + 1) / 10.0, 0, 1)
    state[:, :, 1] = np.clip(state[:, :, 1] / 100.0, -1, 1)
    return state
```

## Workflow

1. Initialize a zero tensor of shape `(board_size, board_size, n_features)`
2. Iterate over all cells; populate resource, unit, and structure channels
3. Sign-encode ownership: +1 for own units, -1 for enemy
4. Log-scale resources to compress dynamic range, then normalize to [0, 1]
5. Clip unit counts to [-1, 1] by dividing by expected max
6. Flatten or pass directly to a CNN policy network

## Key Decisions

- **Log scaling**: resources often span 0 to 10,000+; log2 compresses this to ~13 bits
- **Signed encoding**: lets the network distinguish friend from foe without separate channels
- **Toroidal wrapping**: pad the tensor circularly if using CNNs to respect wrap-around adjacency
- **Channel count**: start minimal (4-6), add channels for cargo, turn number, spawn capacity as needed

## References

- [Reinforcement Learning baseline in Python](https://www.kaggle.com/code/lesamu/reinforcement-learning-baseline-in-python)
