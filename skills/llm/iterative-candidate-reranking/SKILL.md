---
name: llm-iterative-candidate-reranking
description: >
  Narrows a large candidate pool through multiple LLM voting rounds, each presenting a sliding window of candidates plus the current best pick.
---
# Iterative Candidate Reranking

## Overview

When retrieval returns many candidates (25+), a single LLM reranking pass can't compare them all at once due to context limits. Instead, iterate in rounds: each round presents a sliding window of ~8 new candidates plus the surviving best pick from the previous round. The LLM votes for the best, and the winner carries forward. After several rounds, the final survivor is the top-ranked candidate.

## Quick Start

```python
import numpy as np

def iterative_rerank(indices, llm, df, tokenizer, n_rounds=3, window=8):
    """indices: (N_queries, N_candidates) sorted by retrieval score."""
    survivors = indices[:, -1:]  # start with last (worst) as initial survivor

    for i in range(n_rounds):
        # sliding window: 8 new candidates + current survivor
        start = -window * (i + 1) - 1
        end = -window * i - 1
        c_indices = np.concatenate([indices[:, start:end], survivors], axis=1)

        # Build prompts with numbered candidates, ask LLM to pick best
        prompts = build_prompts(df, c_indices, tokenizer)
        responses = llm.generate(prompts, sampling_params)
        choices = parse_choices(responses)  # integer index into c_indices

        # Update survivors
        survivors = np.array([
            c[choice] for choice, c in zip(choices, c_indices)
        ]).reshape(-1, 1)

    return survivors.flatten()
```

## Workflow

1. Sort candidates by initial retrieval score (ascending — worst first)
2. Initialize survivor as the last candidate (will be replaced quickly)
3. Each round: take next window of candidates + survivor, prompt LLM to choose best
4. Update survivor with LLM's choice; slide window toward better candidates
5. Final survivor is the top reranked result

## Key Decisions

- **Window size**: 8-9 fits comfortably in most LLM context windows
- **Number of rounds**: 3 rounds covers ~25 candidates; scale linearly
- **Direction**: Iterate from worst→best so the survivor improves each round
- **Constrained output**: Use logits processor to force single-token choice

## References

- [Qwen14B_Retrieval_Qwen32B_logits-processor-zoo](https://www.kaggle.com/code/jagatkiran/qwen14b-retrieval-qwen32b-logits-processor-zoo)
- [Eedi Qwen32B vllm with logits-processor-zoo](https://www.kaggle.com/code/aerdem4/eedi-qwen32b-vllm-with-logits-processor-zoo)
