---
name: tabular-simulated-annealing-multi-operator
description: Simulated annealing with diverse move operators (translate, rotate, swap, Levy flight, squeeze) and adaptive reheating on stagnation for combinatorial optimization
---

# Simulated Annealing with Multi-Operator Moves

## Overview

Standard simulated annealing with a single perturbation type gets stuck in local optima. Using 10+ move operators — Gaussian translation, rotation, pair swaps, Levy flights, center-directed moves, boundary-focused moves — each randomly selected per step, explores the solution space more effectively. Combined with exponential cooling and adaptive reheating when improvement stalls, this handles diverse combinatorial optimization problems.

## Quick Start

```python
import math, random

def sa_optimize(state, score_fn, operators, T0=3.0, Tmin=1e-6,
                alpha=0.9999, reheat_after=200):
    best = state.copy()
    best_score = cur_score = score_fn(state)
    T = T0
    no_improve = 0
    for step in range(1_000_000):
        op = random.choice(operators)
        candidate = op(state, T / T0)
        new_score = score_fn(candidate)
        delta = new_score - cur_score
        if delta < 0 or random.random() < math.exp(-delta / T):
            state = candidate
            cur_score = new_score
            if new_score < best_score:
                best = state.copy()
                best_score = new_score
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        T *= alpha
        if T < Tmin:
            T = Tmin
        if no_improve > reheat_after:
            T = min(T * 5, T0 * 0.7)
            no_improve = 0
    return best, best_score
```

## Workflow

1. Define state representation and scoring function
2. Implement 5-15 move operators, each scaled by `T/T0` for temperature-aware step sizes
3. Each step: randomly pick an operator, apply, accept/reject via Metropolis criterion
4. Cool exponentially; reheat when no improvement for N steps
5. Return best state seen across all steps

## Key Decisions

- **Operator diversity**: translate, rotate, swap, Levy flight, squeeze, boundary-focus each escape different trap types
- **Temperature-scaled moves**: large moves at high T, fine adjustments at low T
- **Reheat threshold**: 200-600 steps without improvement triggers reheat to `T * 5`
- **Multi-start**: run from multiple initial configs and keep overall best

## References

- [Why Not](https://www.kaggle.com/code/jazivxt/why-not)
- [Santa Claude](https://www.kaggle.com/code/smartmanoj/santa-claude)
