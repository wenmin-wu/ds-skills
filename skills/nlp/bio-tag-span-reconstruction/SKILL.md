---
name: nlp-bio-tag-span-reconstruction
description: >
  Reconstructs named entity spans from BIO token-level tags, handling B/I/O transitions and sentence boundaries.
---
# BIO Tag Span Reconstruction

## Overview

Token-classification NER models output per-token BIO tags (B=begin, I=inside, O=outside). Convert these back into entity text spans by tracking state transitions: B starts a new entity, I continues it, O or another B closes the current span. Essential post-processing for any BERT/transformer NER pipeline.

## Quick Start

```python
def reconstruct_spans(words, tags):
    entities = []
    current = []
    for word, tag in zip(words, tags):
        if tag == "B":
            if current:
                entities.append(" ".join(current))
            current = [word]
        elif tag == "I" and current:
            current.append(word)
        else:
            if current:
                entities.append(" ".join(current))
                current = []
    if current:
        entities.append(" ".join(current))
    return entities

# Usage with BERT NER output:
words = sentence.split()
tags = model.predict(words)  # ["O", "B", "I", "I", "O", ...]
entities = reconstruct_spans(words, tags)
```

## Workflow

1. Get per-token BIO predictions from NER model
2. Iterate through word-tag pairs tracking current entity state
3. On B: close any open entity, start new one
4. On I with open entity: append word to current span
5. On O or end: close current entity and add to results

## Key Decisions

- **BIO vs BIOES**: BIOES adds S(single) and E(end) for finer boundaries; reconstruction logic is similar
- **Subword handling**: Merge subword tokens (##pieces) back into words before reconstruction
- **Confidence filtering**: Optionally skip entities where any token has low prediction confidence
- **Deduplication**: Post-filter extracted entities with Jaccard similarity to remove near-duplicates

## References

- [Coleridge: Matching + BERT NER](https://www.kaggle.com/code/tungmphung/coleridge-matching-bert-ner)
- [Pytorch BERT for Named Entity Recognition](https://www.kaggle.com/code/tungmphung/pytorch-bert-for-named-entity-recognition)
