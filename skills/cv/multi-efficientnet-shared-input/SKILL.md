---
name: cv-multi-efficientnet-shared-input
description: Combine EfficientNetB0..B6 into one Keras model with a shared image input and one sigmoid head per backbone, training all N models in a single fit() call on TPU
---

## Overview

Training N separate CNN backbones (B0 through B6) in sequence wastes TPU idle time and checkpoint bandwidth. A much cleaner pattern: build *one* Keras model with N parallel branches that share the same input tensor, each branch its own EfficientNet backbone and sigmoid head, and N copies of the loss. One `fit()` call trains them all at the same LR schedule on the same batches, and at inference you get N predictions per image for free — perfect for ensembling. Used in the Kaggle SIIM-ISIC Melanoma top solutions to fine-tune 7 EfficientNets in the time of one.

## Quick Start

```python
import tensorflow as tf
import efficientnet.tfkeras as efn

def multi_effnet(img_size, n_nets=7, label_smoothing=0.05):
    inp = tf.keras.Input(shape=(img_size, img_size, 3))
    dummy = tf.keras.layers.Lambda(lambda x: x)(inp)  # forces TF to share
    outputs = []
    for i in range(n_nets):
        Ctor = getattr(efn, f'EfficientNetB{i}')
        x = Ctor(include_top=False, weights='noisy-student',
                 input_shape=(img_size, img_size, 3), pooling='avg')(dummy)
        x = tf.keras.layers.Dense(1, activation='sigmoid', name=f'head_b{i}')(x)
        outputs.append(x)
    model = tf.keras.Model(inp, outputs)
    losses = [tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
              for _ in range(n_nets)]
    model.compile(optimizer='adam', loss=losses,
                  metrics=[tf.keras.metrics.AUC()])
    return model

# Broadcast a single label to one per head
ds = ds.map(lambda img, y: (img, tuple([y] * 7)))
model.fit(ds, epochs=10)
```

## Workflow

1. Build one `Input` layer and pass it through a `Lambda(x: x)` so TF treats the input as shareable
2. Loop over the N backbones, each with `include_top=False, pooling='avg'`
3. Add a per-backbone `Dense(1, sigmoid)` output and collect the list
4. Construct `Model(inp, outputs)` and compile with one loss per output
5. In the tf.data pipeline, map labels to `tuple([y] * N)` so every head sees the same target

## Key Decisions

- **Shared input Lambda**: without the Lambda, some TPU builds refuse to share the input tensor across branches.
- **Same LR for all backbones**: B0 and B6 converge at different rates in theory but in practice a shared schedule works fine with short fine-tuning.
- **Label broadcast, not separate datasets**: one dataset with `tuple(labels * N)` is faster than N datasets.
- **vs. sequential training**: N sequential models = N × data loading + N × optimizer state + N × fit() overhead. Shared-input is usually 2-4× faster end to end.

## References

- [Incredible TPUs - finetune EffNetB0-B6 at once](https://www.kaggle.com/code/cdeotte/incredible-tpus-finetune-effnetb0-b6-at-once)
