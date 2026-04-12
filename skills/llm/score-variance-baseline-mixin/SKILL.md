---
name: llm-score-variance-baseline-mixin
description: Mix a small fraction of plain baseline responses into an adversarial submission to preserve cross-row score variance
---

## Overview

In metrics that reward *disagreement* between LLM judges (e.g. variance or rank-correlation penalties), a perfectly-adversarial submission where every row gets the same derailed score is actually bad — the variance collapses to zero. The fix: keep a 10-15% fraction of rows as plain, un-attacked baselines. The baselines produce a normal score distribution; the adversarial rows cluster at the derailed value. The bimodal mixture has higher variance than either alone, which is exactly what the metric rewards.

## Quick Start

```python
import random

def submission_row(row, rng, baseline_frac=0.10):
    if rng.random() < baseline_frac:
        return plain_essay(row), "baseline"
    else:
        return adversarial_essay(row), "attack"

rng = random.Random(42)
submission = [submission_row(r, rng) for r in test_rows]
```

## Workflow

1. Implement both a `plain_essay` and an `adversarial_essay` function
2. Sweep `baseline_frac` on a held-out slice: measure the competition metric for frac in {0, 0.05, 0.10, 0.15, 0.20}
3. Pick the frac that maximizes the metric — usually 0.08-0.15
4. Use a deterministic seed so the baseline/attack split is reproducible
5. Log which rows were baselines — needed if you need to patch only the attack rows later

## Key Decisions

- **Small fraction**: too much baseline drowns the attack signal; too little kills variance.
- **Random, not block**: random placement prevents the judge from detecting a contiguous "attack region".
- **vs. noise injection**: adding Gaussian noise to scores requires post-processing; baseline-mixing produces real variance via real model outputs.
- **vs. single strategy**: a pure-attack submission often scores worse than a mixed one even on attack-friendly metrics, because modern metrics penalize degenerate distributions.

## References

- [LLMs - You Can't Please Them All competition solutions](https://www.kaggle.com/competitions/llms-you-cant-please-them-all)
