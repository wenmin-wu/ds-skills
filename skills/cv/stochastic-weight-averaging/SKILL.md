---
name: cv-stochastic-weight-averaging
description: Keras callback implementing Stochastic Weight Averaging (SWA) — running mean of model weights over final training epochs
---

## Overview

SWA averages the weights of a model across the final epochs of training. This simple trick often gives a significant boost over picking the single best checkpoint because the average sits in a wider, flatter region of the loss landscape, generalizing better. Unlike checkpoint ensembles, SWA costs nothing at inference (one model, one forward pass).

## Quick Start

```python
import keras

class SWA(keras.callbacks.Callback):
    def __init__(self, filepath, swa_start_epoch):
        super().__init__()
        self.filepath = filepath
        self.swa_start = swa_start_epoch
        self.swa_weights = None

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.swa_start:
            self.swa_weights = self.model.get_weights()
        elif epoch > self.swa_start:
            n = epoch - self.swa_start
            new = self.model.get_weights()
            for i in range(len(self.swa_weights)):
                self.swa_weights[i] = (self.swa_weights[i] * n + new[i]) / (n + 1)

    def on_train_end(self, logs=None):
        self.model.set_weights(self.swa_weights)
        self.model.save_weights(self.filepath)

# Usage
model.fit(..., callbacks=[SWA('swa.h5', swa_start_epoch=10)])
```

## Workflow

1. Train model normally until `swa_start_epoch` (usually 60-80% of total epochs)
2. From that epoch on, maintain a running uniform mean of the weight tensors
3. At the end of training, replace model weights with the running mean
4. Save the SWA weights as the final checkpoint

## Key Decisions

- **Start epoch**: Too early and the running mean includes noisy early weights. Too late and few epochs contribute. Typically 60-80% of total epochs.
- **Constant LR after start**: Best results when LR is held constant (or on a cyclic schedule) after SWA begins, so multiple "good" minima contribute.
- **BatchNorm**: If using BN, run a few forward passes over training data after averaging to recompute running stats (not shown above).
- **vs. checkpoint ensembling**: Same inference cost as one model. No need to load multiple checkpoints.

## References

- [Nested Unet with EfficientNet Encoder](https://www.kaggle.com/code/meaninglesslives/nested-unet-with-efficientnet-encoder)
