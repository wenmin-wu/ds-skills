---
name: cv-smm-spatial-observation-encoding
description: Encode game state as a Super Mini Map (SMM) with separate binary channels for players, ball, and ownership, bit-packed for efficient transfer in RL training
---

# SMM Spatial Observation Encoding

## Overview

For RL agents in spatial games (football, RTS), raw observation vectors (player positions as floats) lose spatial relationships. The Super Mini Map (SMM) renders game state onto a small 2D grid with separate binary channels: one for left team positions, one for right team, one for ball, one for active player. This gives CNN policies a spatial inductive bias while staying compact via bit-packing.

## Quick Start

```python
import numpy as np
from collections import deque

def render_smm(obs, width=96, height=72):
    smm = np.zeros((height, width, 4), dtype=np.uint8)
    def plot(positions, channel):
        for x, y in positions:
            px = int((x + 1) / 2 * (width - 1))
            py = int((y + 0.42) / 0.84 * (height - 1))
            px, py = np.clip(px, 0, width-1), np.clip(py, 0, height-1)
            smm[py, px, channel] = 255
    plot(obs["left_team"], 0)
    plot(obs["right_team"], 1)
    plot([obs["ball"][:2]], 2)
    plot([obs["left_team"][obs["active"]]], 3)
    return smm

# Frame stacking with bit-packing
frames = deque(maxlen=4)
obs = render_smm(raw_obs)
frames.extend([obs] * 4)
stacked = np.concatenate(list(frames), axis=-1)
packed = np.packbits(stacked, axis=-1)
```

## Workflow

1. Define a small grid (72x96 or 48x64) matching the field's aspect ratio
2. Map normalized positions to pixel coordinates with boundary clipping
3. Render each entity type into a separate binary channel
4. Stack 4 consecutive frames for temporal context (16 channels total)
5. Bit-pack the binary arrays for 8x memory reduction during transfer
6. Feed to a CNN policy network (Conv2D → Dense → action logits)

## Key Decisions

- **Resolution**: 72x96 balances spatial precision vs. computation; lower res for faster training
- **Channels**: 4 base channels (left team, right team, ball, active player); add ownership, direction as needed
- **Frame stacking**: 4 frames captures velocity without explicit velocity channels
- **Bit-packing**: reduces observation size 8x for distributed RL data transfer; unpack before inference
- **vs. raw vectors**: SMM gives CNN spatial inductive bias; raw vectors are better for MLP policies

## References

- [GFootball - train SEED RL agent](https://www.kaggle.com/code/piotrstanczyk/gfootball-train-seed-rl-agent)
- [Convolutional Deep-Q learner](https://www.kaggle.com/code/garethjns/convolutional-deep-q-learner)
