---
name: nlp-spearman-correlation-callback
description: >
  Custom Keras callback that evaluates Spearman correlation on validation data each epoch with optional early stopping.
---
# Spearman Correlation Callback

## Overview

For ranking tasks where Spearman correlation is the evaluation metric, build a custom Keras callback that computes per-target Spearman rho on validation data after each epoch. Enables early stopping on the actual competition metric rather than a proxy loss like MSE.

## Quick Start

```python
import numpy as np
from scipy.stats import spearmanr
from tensorflow.keras.callbacks import Callback

class SpearmanCallback(Callback):
    """Evaluate mean Spearman correlation per epoch."""

    def __init__(self, val_data, val_targets, patience=3):
        super().__init__()
        self.val_data = val_data
        self.val_targets = val_targets
        self.patience = patience
        self.best_score = -1
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.val_data, verbose=0)
        scores = []
        for i in range(self.val_targets.shape[1]):
            rho, _ = spearmanr(self.val_targets[:, i], preds[:, i])
            scores.append(rho)
        mean_rho = np.mean(scores)
        print(f"  val_spearman: {mean_rho:.4f}")
        if mean_rho > self.best_score:
            self.best_score = mean_rho
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
```

## Workflow

1. Prepare validation data separately from training generator
2. Instantiate callback with val features and targets
3. Pass callback to `model.fit(callbacks=[spearman_cb])`
4. Monitor `val_spearman` in logs for convergence
5. Combine with `ModelCheckpoint` to save best-scoring weights

## Key Decisions

- **Per-target vs mean**: Compute per-target rho then average; some targets may be noisier
- **Patience**: 2-3 epochs is typical for fine-tuned transformers
- **Batch prediction**: Use full validation set, not batches, for stable Spearman estimates
- **Tie handling**: Add small noise (`1e-7 * np.random.randn`) if targets have many ties

## References

- Google QUEST Q&A Labeling competition (Kaggle)
- Source: [quest-bert-base-tf2-0](https://www.kaggle.com/code/akensert/quest-bert-base-tf2-0)
