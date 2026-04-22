---
name: cv-siamese-pairwise-comparison-head
description: Siamese network head that compares two embeddings via element-wise multiply, add, abs-diff, and squared-diff features for verification tasks
---

# Siamese Pairwise Comparison Head

## Overview

For verification tasks (same/different identity), a siamese network produces two embeddings from shared weights. Instead of a simple distance metric, concatenate four element-wise operations (multiply, add, absolute difference, squared difference) and pass through a small CNN to learn which comparison features matter. This captures richer pair relationships than Euclidean or cosine distance alone.

## Quick Start

```python
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

def build_head(branch_model, mid=32):
    xa = Input(shape=branch_model.output_shape[1:])
    xb = Input(shape=branch_model.output_shape[1:])
    x1 = Lambda(lambda x: x[0] * x[1])([xa, xb])
    x2 = Lambda(lambda x: x[0] + x[1])([xa, xb])
    x3 = Lambda(lambda x: K.abs(x[0] - x[1]))([xa, xb])
    x4 = Lambda(lambda x: K.square(x))(x3)
    x = Concatenate()([x1, x2, x3, x4])
    emb_dim = branch_model.output_shape[1]
    x = Reshape((4, emb_dim, 1))(x)
    x = Conv2D(mid, (4, 1), activation='relu', padding='valid')(x)
    x = Reshape((emb_dim, mid, 1))(x)
    x = Conv2D(1, (1, mid), activation='linear', padding='valid')(x)
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model([xa, xb], x)
```

## Workflow

1. Build a branch model (CNN backbone) producing a fixed-length embedding
2. Create the comparison head with 4 element-wise operations
3. Feed positive pairs (same class) and negative pairs (different class) with binary labels
4. Train end-to-end with binary crossentropy

## Key Decisions

- **4 operations**: multiply captures co-activation, add captures magnitude, abs-diff and squared-diff capture distance — together they span linear and nonlinear comparisons
- **Conv2D head**: learns to weight which comparison features matter per embedding dimension
- **vs cosine/Euclidean**: richer comparison surface, especially when classes have complex intra-class variation
- **Embedding dim**: 64–512 typical; larger dims need more training data

## References

- [Siamese (pretrained) 0.822](https://www.kaggle.com/code/seesee/siamese-pretrained-0-822)
