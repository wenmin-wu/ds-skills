---
name: cv-quadratic-weighted-kappa-callback
description: >
  Custom training callback that computes Quadratic Weighted Kappa on validation data each epoch and checkpoints the best model.
---
# Quadratic Weighted Kappa Callback

## Overview

Quadratic Weighted Kappa (QWK) measures agreement between predicted and true ordinal labels, penalizing larger disagreements quadratically. It's the standard metric for medical grading tasks (diabetic retinopathy, pathology staging) but isn't available as a built-in training metric. This callback computes QWK at epoch end using sklearn, tracks the best score, and saves the model only when QWK improves — replacing loss-based checkpointing with metric-based checkpointing.

## Quick Start

```python
from sklearn.metrics import cohen_kappa_score
import numpy as np

# PyTorch version
def compute_qwk(model, val_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            preds = model(images.to(device)).cpu().numpy()
            all_preds.extend(np.rint(preds).clip(0, 4).flatten())
            all_labels.extend(labels.numpy().flatten())
    return cohen_kappa_score(all_labels, all_preds, weights='quadratic')

# Keras version
class QWKCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data, save_path='best_model.h5'):
        self.val_data = val_data
        self.save_path = save_path
        self.best_kappa = -1

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.val_data
        y_pred = np.rint(self.model.predict(X_val)).astype(int).clip(0, 4)
        kappa = cohen_kappa_score(y_val, y_pred.flatten(), weights='quadratic')
        print(f" - val_kappa: {kappa:.4f}")
        if kappa > self.best_kappa:
            self.best_kappa = kappa
            self.model.save(self.save_path)
```

## Workflow

1. At each epoch end, generate predictions on the validation set
2. Round continuous predictions to nearest integer class
3. Clip to valid class range (e.g., 0-4)
4. Compute QWK using `cohen_kappa_score(weights='quadratic')`
5. Save model if QWK improves over previous best

## Key Decisions

- **Rounding**: Use `np.rint` for nearest-integer; or optimized thresholds for better QWK
- **Clipping**: Essential — predictions outside [0, n_classes-1] crash kappa computation
- **vs loss-based**: QWK can improve while loss plateaus; metric-based checkpointing is more reliable
- **Patience**: Add early stopping based on QWK plateau (e.g., 5 epochs without improvement)

## References

- [APTOS 2019 DenseNet Keras Starter](https://www.kaggle.com/code/xhlulu/aptos-2019-densenet-keras-starter)
- [EfficientNetB5 with Keras (APTOS 2019)](https://www.kaggle.com/code/carlolepelaars/efficientnetb5-with-keras-aptos-2019)
