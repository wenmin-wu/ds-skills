---
name: cv-levenshtein-distance-metric
description: >
  Evaluates image-to-sequence models using mean Levenshtein edit distance between predicted and ground-truth strings.
---
# Levenshtein Distance Metric

## Overview

For sequence generation tasks where output order matters (molecular formulas, OCR, LaTeX rendering), BLEU and accuracy are too coarse. Levenshtein (edit) distance counts the minimum insertions, deletions, and substitutions to transform the predicted string into the ground truth. Lower is better; 0 means exact match. Works at character or token level.

## Quick Start

```python
import Levenshtein
import numpy as np

def levenshtein_score(y_true, y_pred):
    scores = []
    for true, pred in zip(y_true, y_pred):
        scores.append(Levenshtein.distance(true, pred))
    return np.mean(scores)

# Usage:
preds = ["InChI=1S/C6H12O6", "InChI=1S/C2H6O"]
truth = ["InChI=1S/C6H12O6", "InChI=1S/C2H5OH"]
print(levenshtein_score(truth, preds))  # average edit distance
```

## Workflow

1. Generate predicted sequences via greedy/beam search
2. Decode token IDs back to strings (stop at `<eos>`)
3. Compute Levenshtein distance per sample
4. Report mean distance across the dataset

## Key Decisions

- **Normalized vs raw**: Divide by max(len(true), len(pred)) for 0-1 scale; raw is more interpretable
- **Character vs token level**: Character-level for formulas/OCR; token-level for word sequences
- **Library**: `python-Levenshtein` is C-optimized; `editdistance` is an alternative
- **Complementary metrics**: Report exact-match accuracy alongside mean edit distance

## References

- [InChI / Resnet + LSTM with attention / starter](https://www.kaggle.com/code/yasufuminakama/inchi-resnet-lstm-with-attention-starter)
- [InChI / Resnet + LSTM with attention / inference](https://www.kaggle.com/code/yasufuminakama/inchi-resnet-lstm-with-attention-inference)
