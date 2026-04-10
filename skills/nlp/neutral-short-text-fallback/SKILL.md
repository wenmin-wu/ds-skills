---
name: nlp-neutral-short-text-fallback
description: Override model span predictions with full text for neutral sentiment or very short inputs where sub-span extraction is unreliable
domain: nlp
---

# Neutral / Short Text Span Fallback

## Overview

In span extraction tasks, two cases reliably degrade model quality: (1) neutral-sentiment text where the entire text conveys sentiment equally, and (2) very short texts (1-2 words) where extracting a sub-span is meaningless. For both cases, bypass the model and return the full input text. This simple rule-based fallback consistently improves Jaccard scores by 1-3%.

## Quick Start

```python
def predict_span(text, sentiment, model, tokenizer, offsets):
    """Predict span with rule-based fallback."""
    # Fallback 1: neutral sentiment → return full text
    if sentiment == 'neutral':
        return text
    
    # Fallback 2: very short text → return full text
    if len(text.split()) <= 2:
        return text
    
    # Fallback 3: invalid span (end < start) → return full text
    start_idx, end_idx = model_predict(text, sentiment, model, tokenizer)
    if end_idx < start_idx:
        return text
    
    # Normal case: extract span from offsets
    span = ""
    for ix in range(start_idx, end_idx + 1):
        span += text[offsets[ix][0]:offsets[ix][1]]
        if ix + 1 < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:
            span += " "
    return span.strip()
```

## Key Decisions

- **Neutral = full text**: neutral tweets have Jaccard ~0.97 with full text; model predictions are worse
- **Word count threshold**: 2 words is the standard cutoff; single-word inputs are trivially their own span
- **Invalid span guard**: when argmax(end) < argmax(start), the model is uncertain — fall back safely
- **Apply at inference only**: train on all examples including neutral/short for regularization

## References

- Source: [bert-base-uncased-using-pytorch](https://www.kaggle.com/code/abhishek/bert-base-uncased-using-pytorch)
- Competition: Tweet Sentiment Extraction
