---
name: tabular-temporal-session-aggregation
description: >
  Builds user-level features by accumulating statistics across sequential event sessions before each assessment point.
---
# Temporal Session Aggregation

## Overview

In event-sequence datasets (game logs, clickstreams, app sessions), build features by walking through each user's history chronologically and accumulating statistics up to each prediction point. This captures how a user's behavior evolves — accumulated accuracy, attempt counts, session durations, and event type frequencies.

## Quick Start

```python
import numpy as np
from collections import Counter

def aggregate_user_sessions(user_events, assessment_event_codes):
    """Walk through user's events chronologically, build features at each assessment."""
    accumulated_correct = 0
    accumulated_incorrect = 0
    accumulated_actions = 0
    durations = []
    event_counts = Counter()
    features_list = []

    for session in user_events:
        event_counts[session['type']] += 1
        durations.append(session['duration'])
        accumulated_actions += session['n_actions']

        if session['event_code'] in assessment_event_codes:
            features = {
                'accumulated_correct': accumulated_correct,
                'accumulated_incorrect': accumulated_incorrect,
                'accuracy': accumulated_correct / max(1, accumulated_correct + accumulated_incorrect),
                'total_actions': accumulated_actions,
                'duration_mean': np.mean(durations),
                'duration_std': np.std(durations) if len(durations) > 1 else 0,
                'n_sessions': len(durations),
                **{f'type_{k}_count': v for k, v in event_counts.items()},
            }
            features_list.append(features)

        accumulated_correct += session.get('correct', 0)
        accumulated_incorrect += session.get('incorrect', 0)

    return features_list
```

## Workflow

1. Sort events by user and timestamp
2. Walk chronologically per user, accumulating counters
3. At each prediction point, snapshot all accumulated stats as features
4. Include: counts, means, stds, ratios, event-type frequencies
5. Join with user metadata for final feature set

## Key Decisions

- **Leakage**: Only use events BEFORE each assessment — never include the assessment itself
- **Granularity**: Session-level (grouped by session_id) vs event-level (every row)
- **Cold start**: First assessment has zero history — handle with defaults or drop
- **Feature explosion**: Many event types × many aggregations — use feature selection or GBDT importance pruning

## References

- 2019 Data Science Bowl (Kaggle)
- Source: [data-science-bowl-2019-eda-and-baseline](https://www.kaggle.com/code/erikbruin/data-science-bowl-2019-eda-and-baseline)
