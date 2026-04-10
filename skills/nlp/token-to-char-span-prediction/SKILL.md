---
name: nlp-token-to-char-span-prediction
description: Map token-level classifier outputs back to character-level spans via offset mapping, thresholding, and contiguous grouping
domain: nlp
---

# Token-to-Character Span Prediction

## Overview

End-to-end pipeline for extracting text spans from token-level classifiers. Token probabilities are projected onto character positions using offset mappings, thresholded, and grouped into contiguous spans. Used in NER, clinical note annotation, and extractive QA.

## Quick Start

```python
import numpy as np
import itertools

def get_char_probs(texts, predictions, tokenizer):
    """Project per-token probabilities onto character positions."""
    results = [np.zeros(len(t)) for t in texts]
    for i, (text, pred) in enumerate(zip(texts, predictions)):
        enc = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
        for (start, end), p in zip(enc["offset_mapping"], pred):
            results[i][start:end] = p
    return results

def get_spans(char_probs, threshold=0.5):
    """Threshold char probs and group consecutive indices into spans."""
    spans = []
    for probs in char_probs:
        indices = np.where(probs >= threshold)[0]
        groups = [list(g) for _, g in itertools.groupby(
            indices, key=lambda n, c=itertools.count(): n - next(c))]
        spans.append([(min(g), max(g) + 1) for g in groups])
    return spans
```

## Workflow

1. Run token-level classifier (e.g. DeBERTa) → sigmoid probabilities per token
2. Map token probs → character probs via `offset_mapping`
3. Sweep threshold on OOF data (typically 0.45–0.55)
4. Group contiguous above-threshold characters into spans

## Key Decisions

- **Offset mapping**: each token maps to `(char_start, char_end)` — assign token prob to all chars in range
- **Threshold sweep**: 0.01 grid search on OOF predictions maximizes span-level F1
- **Groupby trick**: `itertools.groupby` with counter detects gaps between consecutive indices

## References

- Source: [nbme-deberta-base-baseline-train](https://www.kaggle.com/code/yasufuminakama/nbme-deberta-base-baseline-train)
- Source: [roberta-strikes-back](https://www.kaggle.com/code/theoviel/roberta-strikes-back)
- Competition: NBME - Score Clinical Patient Notes
