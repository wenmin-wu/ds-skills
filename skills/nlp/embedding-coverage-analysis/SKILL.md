---
name: nlp-embedding-coverage-analysis
description: Measure pretrained embedding coverage over dataset vocab and return OOV words sorted by frequency for targeted preprocessing
---

## Overview

Before training with pretrained embeddings, measure how much of your dataset vocabulary actually has a pretrained vector. Low coverage means most real signal is being replaced with zeros/noise. This diagnostic function returns vocab coverage, text coverage (weighted by frequency), and a sorted list of OOV words — letting you target preprocessing at the OOV words that hurt most.

## Quick Start

```python
import operator

def check_coverage(vocab, embeddings_index):
    """Return OOV words sorted by frequency."""
    a = {}
    oov = {}
    k = 0  # count of in-vocab tokens (weighted)
    i = 0  # count of OOV tokens (weighted)
    for word in vocab:
        if word in embeddings_index:
            a[word] = embeddings_index[word]
            k += vocab[word]
        else:
            oov[word] = vocab[word]
            i += vocab[word]
    print(f'Found embeddings for {len(a)/len(vocab):.2%} of vocab')
    print(f'Found embeddings for {k/(k+i):.2%} of all text')
    return sorted(oov.items(), key=operator.itemgetter(1), reverse=True)

# Usage
vocab = build_vocab(train_texts)  # {word: count}
oov = check_coverage(vocab, embeddings_index)
print(oov[:30])  # top 30 most-frequent OOV words
```

## Workflow

1. Build a frequency vocab from the training corpus
2. Call `check_coverage` to get vocab % and text %
3. Inspect the top OOV words — they reveal preprocessing gaps (contractions, numbers, punctuation, misspellings)
4. Apply targeted preprocessing to recover those words
5. Re-run coverage check — aim for > 99% text coverage before training

## Key Decisions

- **Vocab coverage vs text coverage**: Text coverage weights by frequency, so a small list of common OOV words matters more than many rare ones. Optimize text coverage.
- **Target threshold**: Typical good-enough is > 95% vocab and > 99% text. Below that, preprocessing is likely the bottleneck.
- **Ordering**: Return sorted by frequency descending so you fix the biggest wins first.

## References

- [How to: Preprocessing when using embeddings](https://www.kaggle.com/code/christofhenkel/how-to-preprocessing-when-using-embeddings)
