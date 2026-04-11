---
name: nlp-token-map-filtered-topk-spans
description: >
  Filters candidate span indices through a token map to skip special tokens, then cross-products top-k start/end indices with length constraints.
---
# Token-Map Filtered Top-K Spans

## Overview

Extractive QA models output start/end logits over all token positions, including special tokens (CLS, SEP, PAD) and context-only tokens that shouldn't be answer candidates. A token map marks valid answer positions (map value >= 0) vs invalid ones (-1). Filtering through this map before taking top-k indices prevents invalid spans. The cross-product of filtered top-k starts and ends, pruned by ordering and length, gives efficient candidate generation.

## Quick Start

```python
import numpy as np

def get_topk_spans(start_logits, end_logits, token_map, n_best=20, max_len=30):
    """Generate candidate spans from filtered top-k start/end indices.

    Args:
        start_logits: (seq_len,) logits for start positions
        end_logits: (seq_len,) logits for end positions
        token_map: (seq_len,) array; -1 for invalid positions
        n_best: number of top positions to consider
        max_len: maximum span length in tokens
    """
    def topk_filtered(logits):
        # Sort descending, skip position 0 (CLS)
        indices = np.argsort(logits[1:]) + 1
        # Keep only valid answer positions
        indices = indices[token_map[indices] != -1]
        return indices[-n_best:]  # top-k

    starts = topk_filtered(start_logits)
    ends = topk_filtered(end_logits)

    # Cross-product with constraints
    candidates = []
    for s in starts:
        for e in ends:
            if s <= e and (e - s) < max_len:
                score = start_logits[s] + end_logits[e]
                candidates.append((score, int(s), int(e)))
    return sorted(candidates, reverse=True)
```

## Workflow

1. Build token map during preprocessing: valid answer tokens get their word index, others get -1
2. At inference, sort logits and filter through token map
3. Take top-k valid start and end indices
4. Cross-product starts x ends, prune by start < end and max length
5. Score each candidate as sum of start + end logits

## Key Decisions

- **n_best**: 20 is standard; higher values add marginal candidates at quadratic cost
- **max_len**: 30 tokens for short answers; 512+ for long answers
- **Token map source**: Built from tokenizer offset mapping; special tokens and question tokens mapped to -1
- **CLS exclusion**: Always skip position 0 (CLS) from answer candidates

## References

- [BERT Joint Baseline Notebook](https://www.kaggle.com/code/prokaj/bert-joint-baseline-notebook)
