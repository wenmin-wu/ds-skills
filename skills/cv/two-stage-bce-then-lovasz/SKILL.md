---
name: cv-two-stage-bce-then-lovasz
description: Train segmentation with BCE loss first for stable convergence, then fine-tune with Lovasz-hinge on raw logits for IoU-optimal predictions
---

# Two-Stage BCE-then-Lovasz Training

## Overview

BCE loss is smooth and stable for early training but doesn't directly optimize IoU. Lovász-hinge loss is a convex surrogate that directly optimizes IoU but can be unstable from scratch. The two-stage approach trains with BCE first to learn good features, then strips the sigmoid activation and fine-tunes with Lovász-hinge on raw logits. At inference, thresholds must be converted to logit space via the inverse sigmoid.

## Quick Start

```python
from keras.models import load_model, Model

# Stage 1: BCE training
model.compile(loss='binary_crossentropy', optimizer=Adam(1e-3))
model.fit(X_train, y_train, epochs=50,
          callbacks=[ModelCheckpoint('stage1.h5', save_best_only=True)])

# Stage 2: strip sigmoid, switch to Lovász
model = load_model('stage1.h5')
logit_output = model.layers[-1].input  # layer before sigmoid
model2 = Model(model.input, logit_output)
model2.compile(loss=lovasz_loss, optimizer=Adam(1e-4))
model2.fit(X_train, y_train, epochs=30)

# Inference: threshold in logit space
import numpy as np
prob_threshold = 0.5
logit_threshold = np.log(prob_threshold / (1 - prob_threshold))
preds = (model2.predict(X_test) > logit_threshold).astype(np.uint8)
```

## Workflow

1. Train model with BCE loss + sigmoid output until convergence
2. Load best checkpoint, remove final sigmoid activation
3. Recompile with Lovász-hinge loss and lower learning rate
4. Fine-tune on raw logits for 20-50 more epochs
5. At inference, convert probability thresholds to logit space: `logit = log(p / (1-p))`

## Key Decisions

- **Why strip sigmoid**: Lovász-hinge operates on raw logits, not probabilities
- **LR for stage 2**: use 5-10x lower LR than stage 1 to avoid destroying learned features
- **Threshold conversion**: sweep `np.linspace(0.3, 0.7, 31)` in probability space, convert each to logit
- **Stage 1 epochs**: enough for loss plateau — typically 30-60 epochs

## References

- [U-net with simple ResNet Blocks v2 (New loss)](https://www.kaggle.com/code/shaojiaxin/u-net-with-simple-resnet-blocks-v2-new-loss)
