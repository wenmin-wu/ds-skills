---
name: llm-confidence-threshold-fallback
description: >
  Uses primary model predictions only when confidence exceeds a threshold, falling back to a backup ensemble otherwise.
---
# Confidence Threshold Fallback

## Overview

When a primary model (e.g., LLM with RAG) is strong but unreliable on some inputs, set a confidence threshold on its softmax output. Use its prediction when max probability exceeds the threshold; otherwise fall back to a pre-computed backup (e.g., ensemble of smaller models). This combines the strengths of both approaches.

## Quick Start

```python
import numpy as np
from scipy.special import softmax

def predict_with_fallback(primary_logits, backup_preds, threshold=0.4):
    """Use primary model if confident, else fallback."""
    results = []
    for i, logits in enumerate(primary_logits):
        probs = softmax(logits)
        if probs.max() > threshold:
            ranked = np.argsort(-probs)[:3]
            pred = " ".join(["ABCDE"[j] for j in ranked])
        else:
            pred = backup_preds[i]
        results.append(pred)
    return results
```

## Workflow

1. Run primary model (e.g., LLM + retrieval) on all inputs
2. Compute softmax probabilities from logits
3. If max prob > threshold, use primary prediction
4. Otherwise, use backup model's prediction
5. Tune threshold on validation set to maximize MAP@3

## Key Decisions

- **Threshold tuning**: Optimize on validation; 0.3-0.5 typical for 5-class
- **Backup model**: Can be a fine-tuned DeBERTa ensemble or simpler model
- **When to use**: When primary model is accurate but inconsistent
- **Metrics**: Track fallback rate — if >50%, the primary model is too weak

## References

- Kaggle LLM Science Exam (Kaggle)
- Source: [86-2-with-only-270k-articles](https://www.kaggle.com/code/mbanaei/86-2-with-only-270k-articles)
