---
name: tabular-multi-output-auxiliary-targets
description: >
  Neural network with multiple output heads for main target plus auxiliary targets, improving representation learning via shared layers.
---
# Multi-Output Auxiliary Targets

## Overview

Add extra output heads to a neural network that predict related auxiliary targets alongside the main target. The shared hidden layers learn richer representations because they must encode information useful for multiple tasks. Weight auxiliary losses lower than the main loss to keep focus on the primary objective.

## Quick Start

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler

def build_multi_output_model(n_features, aux_dims=[2, 6, 12]):
    inp = Input(shape=(n_features,))
    x = Dense(512, activation='relu')(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # Main target
    main_out = Dense(1, activation='linear', name='main')(x)

    # Auxiliary outputs
    aux_outs = []
    for i, dim in enumerate(aux_dims):
        aux = Dense(dim, activation='linear', name=f'aux_{i}')(x)
        aux_outs.append(aux)

    model = Model(inputs=inp, outputs=[main_out] + aux_outs)

    # Weight auxiliary losses lower
    losses = {'main': 'mse'}
    loss_weights = {'main': 1.0}
    for i in range(len(aux_dims)):
        losses[f'aux_{i}'] = 'mse'
        loss_weights[f'aux_{i}'] = 0.1

    model.compile(optimizer='adam', loss=losses, loss_weights=loss_weights)
    return model
```

## Workflow

1. Identify auxiliary targets available in training data (related properties, labels)
2. Scale auxiliary targets independently (StandardScaler per target)
3. Build shared trunk with separate output heads
4. Set auxiliary loss weights to 0.05-0.2 of main loss
5. Train jointly — shared layers learn from all targets

## Key Decisions

- **Auxiliary weight**: 0.1 is a good start; too high hijacks training away from main target
- **Which auxiliaries**: Must be genuinely related — random targets add noise
- **Scale**: Standardize each auxiliary independently so losses are comparable
- **When to use**: When auxiliary labels are free (already in dataset) and correlated with main target

## References

- Predicting Molecular Properties / CHAMPS Scalar Coupling (Kaggle)
- Source: [keras-neural-net-for-champs](https://www.kaggle.com/code/todnewman/keras-neural-net-for-champs)
