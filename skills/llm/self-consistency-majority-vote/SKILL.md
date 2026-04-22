---
name: llm-self-consistency-majority-vote
description: Aggregate multiple LLM reasoning attempts via majority voting with random jitter tiebreaking and validity filtering
---

# Self-Consistency Majority Vote

## Overview

Self-consistency generates multiple independent reasoning chains for the same problem, then selects the most common answer via majority voting. This exploits the observation that correct reasoning paths converge on the same answer, while errors are random and diverse. Add tiny random jitter to vote counts for deterministic tiebreaking without arbitrary selection.

## Quick Start

```python
from collections import Counter
import random

def majority_vote(answers, default=0, modulo=None):
    counter = Counter()
    for ans in answers:
        try:
            val = int(float(ans))
            counter[val] += 1 + random.random() / 1000
        except (ValueError, TypeError):
            continue

    if not counter:
        return default

    best = max(counter.items(), key=lambda x: x[1])[0]
    return best % modulo if modulo else best

# Generate N responses with temperature > 0
responses = [generate(prompt, temperature=1.0) for _ in range(16)]
answers = [extract_answer(r) for r in responses]
final = majority_vote(answers, default=210, modulo=1000)
```

## Workflow

1. Generate N independent responses (N=8-32) with temperature ≥ 0.7
2. Extract answers from each response (e.g., \boxed{} parsing)
3. Filter invalid answers (non-numeric, out of range)
4. Count occurrences with random jitter for tiebreaking
5. Return the most frequent valid answer, or a default fallback

## Key Decisions

- **N samples**: 16 is a good default; diminishing returns beyond 32
- **Temperature**: 0.7-1.0 for diversity; too low produces identical chains
- **Jitter**: `random.random() / 1000` breaks ties without biasing selection
- **Validity filter**: domain-specific (integer check, range check, modulo)
- **Default**: return a safe fallback when no valid answers exist

## References

- [Deepseek-r1-distill-qwen-7b](https://www.kaggle.com/code/itahiro/deepseek-r1-distill-qwen-7b)
- [[LB 20] QWQ-32B-preview Optimized inference](https://www.kaggle.com/code/mbmmurad/lb-20-qwq-32b-preview-optimized-inference)
