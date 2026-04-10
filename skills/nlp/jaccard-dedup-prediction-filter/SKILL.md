---
name: nlp-jaccard-dedup-prediction-filter
description: >
  Deduplicates extracted entity predictions by filtering out candidates whose Jaccard word-overlap with already-accepted labels exceeds a threshold.
---
# Jaccard Dedup Prediction Filter

## Overview

NER and entity extraction pipelines often produce near-duplicate predictions ("National Health Survey" vs "National Health Survey Data"). Greedy Jaccard deduplication: sort candidates by length, accept each only if its word-level Jaccard similarity with all previously accepted labels is below a threshold. Removes redundant extractions while preserving distinct entities.

## Quick Start

```python
def jaccard_similarity(s1, s2):
    words1 = set(s1.lower().split())
    words2 = set(s2.lower().split())
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0

def dedup_entities(entities, threshold=0.75):
    filtered = []
    for entity in sorted(entities, key=len, reverse=True):
        if not filtered or all(
            jaccard_similarity(entity, kept) < threshold
            for kept in filtered
        ):
            filtered.append(entity)
    return filtered

# Usage:
raw_predictions = ["National Health Survey", "National Health Survey Data", "Census Bureau"]
clean = dedup_entities(raw_predictions, threshold=0.75)
# -> ["National Health Survey Data", "Census Bureau"]
```

## Workflow

1. Collect all raw entity predictions for a document
2. Sort by length (longest first to prefer more complete mentions)
3. Greedily accept each candidate if Jaccard < threshold vs all accepted
4. Output filtered set as pipe-delimited string or list

## Key Decisions

- **Threshold**: 0.5 = strict (keeps more variants); 0.75 = moderate; 1.0 = exact dedup only
- **Sort order**: Longest-first keeps the most informative mention; shortest-first is more conservative
- **Word-level vs char-level**: Word Jaccard is standard for entity dedup; char-level for typo handling
- **Case sensitivity**: Lowercase before comparison to catch capitalization variants

## References

- [Coleridge: Matching + BERT NER](https://www.kaggle.com/code/tungmphung/coleridge-matching-bert-ner)
- [External_Datasets_Matching + MLM](https://www.kaggle.com/code/chienhsianghung/external-datasets-matching-mlm)
