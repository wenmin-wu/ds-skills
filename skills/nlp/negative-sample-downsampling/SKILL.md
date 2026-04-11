---
name: nlp-negative-sample-downsampling
description: >
  Downsamples documents with no entity labels while keeping all positive samples, balancing class distribution in NER training without discarding entity-bearing examples.
---
# Negative Sample Downsampling

## Overview

In NER datasets, most documents contain no entities — in PII detection, 70-90% of essays are clean. Training on all of them wastes compute on easy negatives and biases the model toward predicting O (non-entity). Downsampling negative documents to 20-33% of their original count while keeping all positive documents rebalances the training set. This is simpler than token-level class weighting and more effective — the model sees more entity examples per epoch, improving recall by 2-5%.

## Quick Start

```python
import random

def downsample_negatives(data, keep_ratio=0.2, entity_label='O'):
    """Keep all positive docs, downsample negative docs."""
    positives = []
    negatives = []
    for doc in data:
        labels = doc['labels'] if isinstance(doc['labels'], list) else doc['labels'].tolist()
        if set(labels) != {entity_label}:
            positives.append(doc)
        else:
            negatives.append(doc)

    n_keep = int(len(negatives) * keep_ratio)
    random.shuffle(negatives)
    sampled_negatives = negatives[:n_keep]

    print(f"Positives: {len(positives)}, Negatives: {len(negatives)} -> {n_keep}")
    return positives + sampled_negatives

# Alternative: filter function for HuggingFace datasets
def filter_no_entity(example, keep_ratio=0.2):
    has_entity = set(example['labels']) != {'O'}
    return has_entity or (random.random() < keep_ratio)

dataset = dataset.filter(filter_no_entity)
```

## Workflow

1. Split documents into positive (contains entities) and negative (all O)
2. Keep all positive documents
3. Randomly sample `keep_ratio` of negative documents
4. Combine and shuffle for training

## Key Decisions

- **keep_ratio**: 0.2-0.33 is typical; too low causes false positives on clean documents
- **Per-epoch resampling**: Resample negatives each epoch for diversity
- **vs class weights**: Downsampling is simpler and works at document level; class weights work at token level
- **Validation**: Do NOT downsample validation set — evaluate on the true distribution

## References

- [DeBERTa3base Training](https://www.kaggle.com/code/valentinwerner/915-deberta3base-training)
