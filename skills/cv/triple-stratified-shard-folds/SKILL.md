---
name: cv-triple-stratified-shard-folds
description: Pre-shard TFRecords into N files each balanced along 3 axes (patient, target, image-count), then KFold over file indices for leak-free triple-stratified folds
---

## Overview

Kaggle medical-imaging competitions often need splits stratified along several axes at once: (a) no patient appears in both train and val, (b) class balance held constant, (c) per-patient image-count distribution preserved. Doing this on-the-fly per run is slow and hard to verify. The trick: pre-shard the dataset into 15 TFRecord files where each file already contains a balanced mix along all three axes. At train time a simple `KFold` over file *indices* gives you triple-stratified, leak-free folds with zero runtime stratification logic.

## Quick Start

```python
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf

FOLDS, N_SHARDS, SEED = 5, 15, 42
GCS = 'gs://my-bucket/melanoma-256'

# Each train%.2i.tfrec contains a balanced mix along patient/target/image-count
skf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

for fold, (idxT, idxV) in enumerate(skf.split(np.arange(N_SHARDS))):
    files_train = tf.io.gfile.glob(
        [f'{GCS}/train{x:02d}*.tfrec' for x in idxT])
    files_valid = tf.io.gfile.glob(
        [f'{GCS}/train{x:02d}*.tfrec' for x in idxV])
    train_ds = tf.data.TFRecordDataset(files_train).map(parse_fn)
    # Patients never cross folds because each shard is one patient-group
    # Target + image-count are balanced by construction when sharding
```

## Workflow

1. Offline: group training rows by patient → distribute groups into N shards greedily balancing target count and mean image-count per shard
2. Write each shard as one TFRecord file named `train00.tfrec`, `train01.tfrec`, ...
3. At train time, run `KFold(n_splits=F)` over `np.arange(N)` to get shard-index splits
4. Glob the train/val shard filenames for each fold and build `tf.data` pipelines
5. Validate: assert no patient_id appears in both train and val shard sets (one-time sanity check)

## Key Decisions

- **N shards = 15 is flexible**: works for 3-fold (5 shards each) and 5-fold (3 shards each) without re-sharding.
- **Offline sharding is reusable**: write once, use across all experiments; all folds now run identically.
- **Balance greedily, not optimally**: greedy assignment (largest-first) is fast and gets within ~1% of perfect balance.
- **vs. GroupKFold at runtime**: runtime GroupKFold still loads every row to check the group — shard-file KFold reads only the chosen files.

## References

- [Triple Stratified KFold with TFRecords](https://www.kaggle.com/code/cdeotte/triple-stratified-kfold-with-tfrecords)
