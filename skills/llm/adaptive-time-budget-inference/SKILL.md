---
name: llm-adaptive-time-budget-inference
description: Dynamically reduce max_tokens and batch size as wall-clock time approaches a cutoff to ensure all inputs get processed
---

# Adaptive Time Budget Inference

## Overview

In time-limited settings (competitions, APIs with timeout), LLM inference must balance quality (more tokens, more samples) against the risk of timeout. Monitor wall-clock time and progressively reduce max_tokens and batch size as the deadline approaches. Precompute a cutoff schedule so the check is a simple comparison, not a complex calculation.

## Quick Start

```python
import time
import numpy as np

MAX_TOKENS = 16384
MAX_BATCH = 16
TIME_LIMIT = 4 * 60 * 60 + 45 * 60  # 4h45m

start_time = time.time()
cutoff_time = start_time + TIME_LIMIT

def get_params():
    elapsed_frac = (time.time() - start_time) / TIME_LIMIT
    if elapsed_frac > 0.9:
        return MAX_TOKENS // 3, MAX_BATCH // 3  # emergency mode
    elif elapsed_frac > 0.7:
        return 2 * MAX_TOKENS // 3, 2 * MAX_BATCH // 3  # reduced
    return MAX_TOKENS, MAX_BATCH  # full budget

def predict(question):
    if time.time() > cutoff_time:
        return default_answer
    max_tokens, batch_size = get_params()
    # ... generate with adjusted params ...
```

## Workflow

1. Record start time and compute absolute cutoff
2. Define budget tiers (full → reduced → emergency → default)
3. Before each generation, check elapsed fraction
4. Reduce max_tokens and num_samples as time pressure increases
5. Return safe default if past cutoff

## Key Decisions

- **Tiers**: 3 tiers (100%/67%/33%) is practical; more tiers add complexity without benefit
- **What to reduce**: max_tokens has the biggest time impact; num_samples second
- **Default answer**: competition-specific fallback (e.g., median, mode of training labels)
- **Buffer**: set cutoff 5-15 minutes before actual deadline for safety margin

## References

- [Deepseek-r1-distill-qwen-7b](https://www.kaggle.com/code/itahiro/deepseek-r1-distill-qwen-7b)
- [AIMO 2: deepseek-r1-distill-qwen-7b-awq](https://www.kaggle.com/code/yekenot/aimo-2-deepseek-r1-distill-qwen-7b-awq)
