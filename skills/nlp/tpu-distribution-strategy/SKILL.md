---
name: nlp-tpu-distribution-strategy
description: >
  Auto-detects TPU vs CPU/GPU at runtime and wraps model construction in the appropriate TensorFlow distribution strategy with scaled batch size.
---
# TPU Distribution Strategy

## Overview

Kaggle and cloud environments may or may not have TPUs available. Instead of maintaining separate code paths, auto-detect the accelerator at startup and select the correct TF distribution strategy. Scale batch size by the number of replicas so each device gets a consistent local batch.

## Quick Start

```python
import tensorflow as tf

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.get_strategy()  # CPU or single GPU

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

with strategy.scope():
    model = build_model()
    model.compile(optimizer="adam", loss="binary_crossentropy")
```

## Workflow

1. Attempt to resolve a TPU cluster; catch `ValueError` if none exists
2. If TPU found, connect and initialize; create `TPUStrategy`
3. Otherwise fall back to default strategy (mirrors single-device behavior)
4. Scale batch size by `num_replicas_in_sync` (8 for TPU v3-8)
5. Build and compile model inside `strategy.scope()`

## Key Decisions

- **Batch scaling**: Multiply base batch size by replica count to maintain effective batch size
- **Mixed precision**: TPUs use bfloat16 natively; no manual mixed-precision setup needed
- **Data pipeline**: Use `tf.data` with `drop_remainder=True` for TPU (requires fixed shapes)
- **Scope boundary**: Only model creation and compilation go inside `strategy.scope()`

## References

- [Jigsaw TPU: XLM-Roberta](https://www.kaggle.com/code/xhlulu/jigsaw-tpu-xlm-roberta)
- [Deep Learning For NLP: Zero To Transformers & BERT](https://www.kaggle.com/code/tanulsingh077/deep-learning-for-nlp-zero-to-transformers-bert)
