---
name: cv-class-balanced-epoch-undersampling
description: Dynamically undersample majority class each epoch with per-class keep probabilities for stochastic balance
---

## Overview

Instead of fixed undersampling (which discards data permanently), resample at the start of each epoch using per-class keep probabilities. Negative samples are randomly dropped with probability p, so the model sees different negative subsets each epoch while maintaining approximate class balance. This preserves data diversity across epochs while controlling imbalance.

## Quick Start

```python
import numpy as np

class BalancedGenerator:
    def __init__(self, ids, labels, batch_size, keep_probs=None):
        self.ids = ids
        self.labels = labels
        self.batch_size = batch_size
        self.keep_probs = keep_probs or {0: 0.35, 1: 0.5}

    def on_epoch_end(self):
        keep_prob = self.labels.map(self.keep_probs)
        keep = keep_prob > np.random.rand(len(keep_prob))
        self.indices = np.where(keep)[0]
        np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, idx):
        batch_ids = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self._load_batch(batch_ids)
```

## Workflow

1. Define keep probabilities per class (e.g., negative=0.35, positive=0.5)
2. At each epoch start, draw random numbers for all samples
3. Keep sample if random draw < its class keep probability
4. Shuffle surviving indices and train on this subset
5. Next epoch gets a different random subset

## Key Decisions

- **Keep probabilities**: Set lower for majority class. 0.35 for negatives with 0.5 for positives gives ~1.4:1 ratio from a 6:1 imbalanced set.
- **vs. fixed undersampling**: Fixed discards data permanently. Stochastic sees all data over multiple epochs.
- **vs. oversampling**: No duplicate samples, avoids overfitting to minority class. Works better with augmentation.

## References

- [RSNA InceptionV3 Keras](https://www.kaggle.com/code/akensert/rsna-inceptionv3-keras-tf1-14-0)
