---
name: cv-frame-differencing-temporal-encoding
description: Encode motion and velocity by computing per-channel pixel differences between consecutive frames instead of stacking raw frames for RL visual observations
---

# Frame Differencing Temporal Encoding

## Overview

Frame stacking (concatenating the last N frames) is the standard way to give RL agents temporal information from visual observations. Frame differencing is a lighter alternative: subtract the previous frame from the current one to produce a motion-only image. Moving objects appear as non-zero pixels while static backgrounds cancel out, giving the network explicit velocity signals without doubling the input channels.

## Quick Start

```python
import numpy as np
from collections import deque

class FrameDiffWrapper:
    def __init__(self, env, n_channels=4):
        self.env = env
        self.n_channels = n_channels
        self.buffer = deque(maxlen=2)

    def reset(self):
        obs = self.env.reset()
        frame = obs / 255.0
        self.buffer.append(frame)
        self.buffer.append(np.zeros_like(frame))
        return self._diff()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.buffer.append(obs / 255.0)
        return self._diff(), reward, done, info

    def _diff(self):
        diff = np.empty_like(self.buffer[1])
        for c in range(diff.shape[-1]):
            diff[..., c] = self.buffer[1][..., c] - self.buffer[0][..., c]
        return diff
```

## Workflow

1. Maintain a deque of the last 2 frames (normalized to [0, 1])
2. At each step, compute per-channel difference: `current - previous`
3. Feed the difference image to the policy network instead of stacked frames
4. Moving objects have large positive/negative values; static regions are ~0
5. Optionally combine: stack one raw frame + one diff frame for position + velocity

## Key Decisions

- **Diff vs. stack**: diff uses half the channels (1 vs. 4) with comparable performance for motion-centric tasks
- **Normalization**: normalize to [0, 1] before differencing to keep values in [-1, 1]
- **Combined mode**: raw frame + diff frame gives both position and velocity — best of both worlds
- **Multiple diffs**: stack 2-3 consecutive diffs for acceleration information (diminishing returns)
- **Sparse motion**: in games with few moving objects, diff frames are mostly zero — use sparse convolutions or skip connections

## References

- [Convolutional Deep-Q learner](https://www.kaggle.com/code/garethjns/convolutional-deep-q-learner)
