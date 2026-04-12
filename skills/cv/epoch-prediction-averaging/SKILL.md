---
name: cv-epoch-prediction-averaging
description: Collect test predictions each epoch via callback and combine with exponentially increasing weights favoring later epochs
---

## Overview

Instead of using only the final-epoch model for inference, collect test predictions at the end of every epoch and combine them with exponentially increasing weights. Later epochs get more weight because the model improves over training. This acts as a free ensemble across training checkpoints without saving multiple model files.

## Quick Start

```python
import numpy as np
from keras.callbacks import Callback

class PredictionCheckpoint(Callback):
    def __init__(self, test_generator, test_len):
        self.test_generator = test_generator
        self.test_len = test_len
        self.test_predictions = []

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.test_generator)[:self.test_len]
        self.test_predictions.append(preds)

# After training
weights = [2 ** i for i in range(len(cb.test_predictions))]  # 1, 2, 4, 8, ...
final_preds = np.average(cb.test_predictions, axis=0, weights=weights)
```

## Workflow

1. Create a callback that runs `model.predict` on test data at each epoch end
2. Store predictions in a list indexed by epoch
3. After training completes, define exponential weights: `[2^0, 2^1, ..., 2^(n-1)]`
4. Compute weighted average across all epoch predictions
5. Use the averaged predictions for submission

## Key Decisions

- **Weight scheme**: Exponential (1, 2, 4, 8...) emphasizes later epochs. Linear (1, 2, 3...) is gentler. Skip first few epochs if early predictions are noisy.
- **Memory**: Storing predictions for all epochs uses `n_epochs * n_samples * n_classes` memory. For large test sets, keep only the last K epochs.
- **vs. checkpoint ensemble**: This avoids saving/loading multiple model files. Trade-off: requires test inference at every epoch during training.

## References

- [RSNA InceptionV3 Keras](https://www.kaggle.com/code/akensert/rsna-inceptionv3-keras-tf1-14-0)
