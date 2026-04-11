---
name: cv-ordinal-multilabel-encoding
description: >
  Encodes ordinal classes as cumulative binary labels (class N activates labels 0..N), enabling sigmoid + BCE training for ordinal regression.
---
# Ordinal Multilabel Encoding

## Overview

Standard classification treats ordinal classes (severity grades 0-4) as unrelated categories, ignoring their natural order. Ordinal multilabel encoding converts class K into K+1 binary labels where all labels up to K are activated. For example, class 3 becomes [1,1,1,1,0]. Training with sigmoid + BCE per label teaches the model that higher classes imply all lower classes. Decoding is simply summing active labels minus one.

## Quick Start

```python
import numpy as np

def ordinal_encode(labels, n_classes=5):
    """Convert ordinal labels to cumulative binary encoding.

    Args:
        labels: (N,) integer labels in [0, n_classes-1]
        n_classes: number of ordinal classes
    Returns:
        (N, n_classes) binary array
    """
    encoded = np.zeros((len(labels), n_classes), dtype=np.float32)
    for i, label in enumerate(labels):
        encoded[i, :label + 1] = 1.0
    return encoded

def ordinal_decode(preds, threshold=0.5):
    """Decode sigmoid predictions back to ordinal labels."""
    return (preds > threshold).astype(int).sum(axis=1) - 1

# Training
y_encoded = ordinal_encode(y_train, n_classes=5)
model.compile(loss='binary_crossentropy', optimizer='adam')  # sigmoid output
model.fit(X_train, y_encoded)

# Inference
y_pred = ordinal_decode(model.predict(X_test))
```

## Workflow

1. Encode ordinal labels as cumulative binary vectors
2. Use sigmoid activation (not softmax) on the output layer
3. Train with binary cross-entropy loss per label
4. At inference, threshold each sigmoid output and sum active labels

## Key Decisions

- **vs softmax**: Softmax ignores ordinal structure; this encoding enforces monotonicity
- **Threshold**: 0.5 is default; optimize on validation with QWK or other ordinal metric
- **n_classes**: Equals the number of ordinal levels (e.g., 5 for grades 0-4)
- **Alternative**: Frank & Hall method trains K-1 binary classifiers independently

## References

- [APTOS 2019 DenseNet Keras Starter](https://www.kaggle.com/code/xhlulu/aptos-2019-densenet-keras-starter)
