---
name: llm-weighted-attack-strategy-sampling
description: Sample from a pool of adversarial prompt strategies with per-strategy probability weights to hedge across judge models
---

## Overview

No single adversarial prompt wins against all LLM judges — Claude resists counting traps, Gemini resists metadata injection, GPT resists self-ID prompts. Instead of picking one, maintain a pool of attack strategies and sample one per row using `random.choices` with weights learned from a small validation set. The submission becomes a stochastic mixture that dominates any single strategy because it always has some coverage against the judge du jour.

## Quick Start

```python
import random

STRATEGIES = {
    "counting_trap":    (0.35, build_counting_trap),
    "metadata_inject":  (0.25, build_metadata_inject),
    "few_shot_inject":  (0.20, build_few_shot_inject),
    "score_variance":   (0.10, build_score_variance),
    "plain_essay":      (0.10, build_plain_essay),   # baseline to preserve distribution
}

names   = list(STRATEGIES.keys())
weights = [w for w, _ in STRATEGIES.values()]
builders = {k: b for k, (_, b) in STRATEGIES.items()}

def attack(row, rng=random.Random()):
    strategy = rng.choices(names, weights=weights, k=1)[0]
    return builders[strategy](row), strategy
```

## Workflow

1. Implement 4-7 distinct attack strategies as pure functions `(row) -> essay_text`
2. Run each strategy alone on a small validation set and record the judge-score distribution
3. Set initial weights proportional to per-strategy validation reward
4. Always include a small (5-15%) weight for a plain/baseline strategy so variance doesn't collapse
5. Log the sampled strategy per row — required for post-hoc analysis and weight refinement

## Key Decisions

- **Weighted, not uniform**: strong strategies should appear more often, but never 100% — judges adapt.
- **Include a baseline**: a nonzero probability of plain essays preserves score variance, which is what the competition metric rewards.
- **Seed the RNG per submission, not per row**: reproducibility matters when debugging regressions.
- **vs. ensembling attacks in one prompt**: concatenating all strategies into every essay triggers length limits and cross-interference.

## References

- [LLMs - You Can't Please Them All competition solutions](https://www.kaggle.com/competitions/llms-you-cant-please-them-all)
