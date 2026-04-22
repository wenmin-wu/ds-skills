---
name: tabular-hierarchical-rule-engine
description: Two-level group-then-pattern dispatch for game AI agents where groups filter by game state and ordered patterns within a group fire the first matching action
---

# Hierarchical Rule Engine

## Overview

For heuristic game AI agents, a flat if-else chain becomes unmaintainable past ~20 rules. A two-level dispatch — groups that test broad game state (attacking, defending, set piece), then ordered patterns within the matching group — scales to hundreds of rules while staying readable. Each pattern is a dict with `environment_fits` (predicate) and `get_action` (callback), evaluated in priority order.

## Quick Start

```python
def make_pattern(predicate, action_fn):
    return lambda obs, x, y: {
        "environment_fits": predicate,
        "get_action": action_fn,
    }

def make_group(group_predicate, patterns_fn):
    return lambda obs, x, y: {
        "environment_fits": group_predicate,
        "get_memory_patterns": patterns_fn,
    }

groups = [attack_group, defend_group, set_piece_group, fallback_group]

def get_action(obs, x, y):
    for make_grp in groups:
        grp = make_grp(obs, x, y)
        if grp["environment_fits"](obs, x, y):
            for make_pat in grp["get_memory_patterns"](obs, x, y):
                pat = make_pat(obs, x, y)
                if pat["environment_fits"](obs, x, y):
                    return pat["get_action"](obs, x, y)
    return default_action
```

## Workflow

1. Define groups by broad game state (ball ownership, field zone, game mode)
2. Within each group, define patterns ordered by priority (most specific first)
3. Each pattern has a predicate (`environment_fits`) and action callback
4. At each timestep, find the first matching group, then first matching pattern
5. Always include a fallback group/pattern to avoid returning None

## Key Decisions

- **Group ordering**: most common states first for performance; most critical (set pieces) first for correctness
- **Pattern priority**: specific patterns (corner kick near goal) before general ones (has ball)
- **Shared state**: pass a mutable obs dict so patterns can set flags for downstream patterns
- **vs. decision trees**: rule engine is hand-crafted but transparent; use when domain expertise > data
- **Extensibility**: adding a new behavior = adding one pattern dict to the right group

## References

- [GFootball with Memory Patterns](https://www.kaggle.com/code/yegorbiryukov/gfootball-with-memory-patterns)
