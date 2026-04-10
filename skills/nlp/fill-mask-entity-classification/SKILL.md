---
name: nlp-fill-mask-entity-classification
description: >
  Uses a fine-tuned masked language model to classify candidate entity spans by comparing fill-mask probabilities for positive vs negative marker tokens.
---
# Fill-Mask Entity Classification

## Overview

Instead of training a separate classifier for entity type detection, leverage masked language modeling: replace a candidate entity span with a [MASK] token, then compare the model's fill-mask probabilities for a positive marker (e.g., "$") vs a negative marker (e.g., "#"). If the positive marker is more likely, the span is classified as an entity. Fine-tune the MLM on domain text where markers replace known entities vs non-entities.

## Quick Start

```python
from transformers import pipeline

ENTITY_SYMBOL = "$"
NON_ENTITY_SYMBOL = "#"

mlm = pipeline("fill-mask", model="fine-tuned-mlm-model", device=0)

def classify_candidates(sentences_with_mask):
    results = mlm(sentences_with_mask, targets=[f" {ENTITY_SYMBOL}", f" {NON_ENTITY_SYMBOL}"])
    predictions = []
    for (pos_result, neg_result), phrase in zip(results, candidate_phrases):
        if pos_result["score"] > neg_result["score"] * 2:
            predictions.append(phrase)
    return predictions
```

## Workflow

1. Fine-tune an MLM where "$" replaces entity mentions and "#" replaces non-entity noun phrases
2. At inference, extract candidate noun phrases from text (capitalization heuristics, NP chunking)
3. Replace each candidate with [MASK] in its sentence context
4. Run fill-mask and compare probabilities of "$" vs "#"
5. Accept candidates where entity marker probability dominates (e.g., 2x threshold)

## Key Decisions

- **Marker tokens**: Use single special tokens added to vocabulary; avoid common words
- **Threshold ratio**: 2x is typical; higher = more precision, lower = more recall
- **Candidate extraction**: Capitalized noun phrases, regex patterns, or NP chunker
- **Batching**: Process multiple masked sentences in batch for GPU efficiency

## References

- [External_Datasets_Matching + MLM](https://www.kaggle.com/code/chienhsianghung/external-datasets-matching-mlm)
