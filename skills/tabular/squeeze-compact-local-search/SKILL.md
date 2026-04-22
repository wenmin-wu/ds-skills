---
name: tabular-squeeze-compact-local-search
description: Three-stage packing refinement — uniform squeeze toward centroid, greedy compaction per object, then multi-directional local search — to tighten solutions after metaheuristic optimization
---

# Squeeze-Compact-Local Search Pipeline

## Overview

After simulated annealing or another metaheuristic finds a feasible packing, three refinement stages can squeeze out additional density: (1) **Squeeze** — uniformly scale all objects toward the centroid until overlap occurs, (2) **Compact** — greedily move each object toward the center, accepting if no collision, (3) **Local search** — try small translations and rotations in 8 directions at decreasing step sizes. Run in sequence; each stage feeds its output to the next.

## Quick Start

```python
def squeeze(state, score_fn, overlap_fn, steps=100):
    center = state.centroid()
    for scale in [1 - i * 0.0005 for i in range(steps)]:
        trial = state.scale_toward(center, scale)
        if not overlap_fn(trial):
            state = trial
        else:
            break
    return state

def compact(state, score_fn, overlap_fn, rounds=50):
    center = state.centroid()
    for _ in range(rounds):
        for i in range(len(state)):
            direction = center - state.position(i)
            step = direction * 0.01
            trial = state.move(i, step)
            if not overlap_fn(trial):
                state = trial
    return state

def local_search(state, score_fn, overlap_fn, rounds=80):
    dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)]
    for step_size in [0.1, 0.05, 0.01, 0.005]:
        for _ in range(rounds):
            for i in range(len(state)):
                for dx, dy in dirs:
                    trial = state.move(i, (dx*step_size, dy*step_size))
                    if not overlap_fn(trial) and score_fn(trial) < score_fn(state):
                        state = trial
    return state

result = local_search(compact(squeeze(sa_result, score, overlap), score, overlap), score, overlap)
```

## Workflow

1. **Squeeze**: scale all objects toward centroid with decreasing factor until first overlap
2. **Compact**: for each object, greedily step toward center if no collision
3. **Local search**: 8-directional moves + rotation at decreasing step sizes
4. Accept moves only if feasible and score improves (or stays same)
5. Chain: `squeeze → compact → local_search`, optionally repeat

## Key Decisions

- **Squeeze step**: 0.0005 per step is conservative; larger steps risk missing the tight boundary
- **Compact rounds**: 50 is usually enough for convergence
- **Local search granularity**: start coarse (0.1), end fine (0.005) — progressive refinement
- **Corner-first**: prioritize objects touching the bounding box boundary for bigger gains

## References

- [Santa Claude](https://www.kaggle.com/code/smartmanoj/santa-claude)
