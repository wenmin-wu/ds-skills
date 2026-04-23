---
name: cv-chunked-csv-image-generator
description: Memory-efficient Keras generator that streams sharded CSV files in chunks, renders strokes to images on-the-fly, and yields batches for training on datasets too large for memory
---

# Chunked CSV Image Generator

## Overview

When training on millions of images stored as raw data in CSV files (stroke coordinates, pixel arrays), loading everything into memory is infeasible. A streaming generator reads sharded CSVs in chunks, converts each chunk to images on-the-fly, and yields (X, y) batches. Random shard permutation ensures epoch-level shuffling without random access to the full dataset.

## Quick Start

```python
import pandas as pd
import numpy as np
import json

def image_generator(shard_ids, batch_size, size=64, n_classes=340):
    while True:
        for k in np.random.permutation(shard_ids):
            filename = f"train_shard_{k}.csv.gz"
            for chunk in pd.read_csv(filename, chunksize=batch_size):
                chunk["drawing"] = chunk["drawing"].apply(json.loads)
                X = np.zeros((len(chunk), size, size, 1), dtype=np.float32)
                for i, strokes in enumerate(chunk["drawing"]):
                    X[i, :, :, 0] = render_strokes(strokes, size)
                X = X / 255.0
                y = np.eye(n_classes)[chunk["label"].values]
                yield X, y

train_gen = image_generator(range(100), batch_size=256)
model.fit(train_gen, steps_per_epoch=800, epochs=50)
```

## Workflow

1. Pre-shard the dataset into N compressed CSV files (see deterministic-hash-partitioning)
2. At each epoch, randomly permute shard order for shuffling
3. Read each shard in chunks of `batch_size` rows
4. Parse and render stroke data to images within the chunk
5. Yield (images, one-hot labels) as a batch
6. Loop infinitely — Keras handles epoch boundaries via `steps_per_epoch`

## Key Decisions

- **Chunk size = batch size**: simplest; use 2x batch size if rendering is the bottleneck (pre-render ahead)
- **Shard permutation**: shuffles at shard granularity; within-shard order is pre-shuffled during sharding
- **Compression**: gzip adds ~20% read overhead but 3-5x smaller files — worth it for I/O-bound training
- **Prefetching**: wrap with `tf.data.Dataset.from_generator().prefetch(2)` for GPU pipeline overlap
- **Multi-worker**: use Keras `workers=4, use_multiprocessing=True` if rendering is CPU-bound

## References

- [Greyscale MobileNet [LB=0.892]](https://www.kaggle.com/code/gaborfodor/greyscale-mobilenet-lb-0-892)
