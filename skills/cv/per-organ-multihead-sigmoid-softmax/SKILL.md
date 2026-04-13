---
name: cv-per-organ-multihead-sigmoid-softmax
description: Single CNN backbone with one shallow Dense neck per organ and mixed sigmoid (binary) + softmax (multi-class severity) heads, trained with a dict of losses so each organ is calibrated independently while sharing visual features
---

## Overview

Multi-organ trauma classification has a heterogeneous label structure: some organs are binary (injured / not), others have ordered severity grades (healthy / low / high). The naive answer — one big sigmoid head with all classes flattened — destroys the mutual exclusivity inside each grade group and trains every label to compete with every other label. The right structure is one shared backbone, a tiny per-organ "neck" Dense layer, and a head whose activation matches the label semantics: sigmoid for binary organs, softmax for severity-graded ones. Keras `compile(loss={...})` accepts a dict mapping head names to losses, so each head gets its correct loss without hand-rolling.

## Quick Start

```python
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy

x = GlobalAveragePooling2D()(backbone.output)

necks = {n: Dense(32, activation='silu', name=f'{n}_neck')(x)
         for n in ['bowel', 'extra', 'liver', 'kidney', 'spleen']}

outs = [
    Dense(1, activation='sigmoid', name='bowel')(necks['bowel']),
    Dense(1, activation='sigmoid', name='extra')(necks['extra']),
    Dense(3, activation='softmax', name='liver')(necks['liver']),
    Dense(3, activation='softmax', name='kidney')(necks['kidney']),
    Dense(3, activation='softmax', name='spleen')(necks['spleen']),
]

model = Model(backbone.inputs, outs)
model.compile(
    optimizer='adam',
    loss={
        'bowel':  BinaryCrossentropy(label_smoothing=0.05),
        'extra':  BinaryCrossentropy(label_smoothing=0.05),
        'liver':  CategoricalCrossentropy(label_smoothing=0.05),
        'kidney': CategoricalCrossentropy(label_smoothing=0.05),
        'spleen': CategoricalCrossentropy(label_smoothing=0.05),
    },
)
```

## Workflow

1. Pick the smallest "neck" width that still trains (32 is usually plenty) — smaller necks force the backbone to do the work
2. Use sigmoid heads for binary organs and softmax heads for ordered/multi-class organs in the same model
3. Pass a dict to `compile(loss=...)` matching the head names — Keras auto-routes per-output losses
4. Apply uniform `label_smoothing=0.05` across all heads to prevent any one organ from collapsing onto a 0/1 saturated prediction
5. At inference, concatenate head outputs in the order the submission expects (use `cv-multihead-softmax-to-flat-submission` patterns)

## Key Decisions

- **Per-organ neck, not shared head**: a shared head forces every organ to use the same projection of backbone features and underperforms on rare classes.
- **Sigmoid + softmax in the same model**: forcing everything to sigmoid breaks softmax's mutual-exclusivity guarantee for severity grades; forcing to softmax breaks the binary semantics.
- **`silu` over `relu` in the neck**: smoother gradients on a tiny 32-unit layer; the difference is small but consistently positive.
- **Label smoothing = 0.05, not 0.1**: medical labels are clean enough that aggressive smoothing hurts; 0.05 is the sweet spot for log-loss metrics.
- **Don't share weights between necks**: the whole point is per-organ specialization on a shared visual representation.

## References

- [RSNA-ATD: CNN [TPU][Train]](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection)
