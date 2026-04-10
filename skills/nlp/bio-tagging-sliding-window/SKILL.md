---
name: nlp-bio-tagging-sliding-window
description: >
  Splits long documents into overlapping fixed-length windows with BIO NER tags for BERT token classification on sequences exceeding max length.
---
# BIO Tagging Sliding Window

## Overview

BERT token classifiers have a fixed max length (typically 512 tokens). For long documents, split sentences into overlapping windows so no entity is cut in half. Tag each window independently with BIO labels, then merge results across windows. The overlap ensures entities near window boundaries are captured by at least one window.

## Quick Start

```python
MAX_LENGTH = 400
OVERLAP = 100

def shorten_sentences(sentences):
    short = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) > MAX_LENGTH:
            for start in range(0, len(words), MAX_LENGTH - OVERLAP):
                short.append(" ".join(words[start:start + MAX_LENGTH]))
        else:
            short.append(sentence)
    return short

def tag_sentence(sentence, labels):
    words = sentence.split()
    tags = ["O"] * len(words)
    for label in labels:
        label_words = label.split()
        for i in range(len(words) - len(label_words) + 1):
            if words[i:i+len(label_words)] == label_words:
                tags[i] = "B"
                for j in range(i+1, i+len(label_words)):
                    tags[j] = "I"
    return list(zip(words, tags))
```

## Workflow

1. Split long sentences into windows of `MAX_LENGTH` words with `OVERLAP` overlap
2. Assign BIO tags to each window independently using known entity labels
3. Train BERT token classifier on the windowed data
4. At inference, run model on each window and merge predictions
5. Reconstruct entity spans from merged BIO tags, deduplicating across windows

## Key Decisions

- **MAX_LENGTH**: 300-400 words leaves room for special tokens after tokenization
- **OVERLAP**: 50-150 words; larger overlap = better boundary coverage but more data
- **Merge strategy**: Union of entities from all windows; deduplicate with Jaccard similarity
- **Negative sampling**: Only include negative windows containing domain keywords to reduce imbalance

## References

- [Pytorch BERT for Named Entity Recognition](https://www.kaggle.com/code/tungmphung/pytorch-bert-for-named-entity-recognition)
