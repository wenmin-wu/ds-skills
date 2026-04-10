---
name: tabular-xgb-gpu-batch-iterator
description: Use XGBoost DeviceQuantileDMatrix with a custom batch iterator to train on large datasets without exhausting GPU memory
domain: tabular
---

# XGBoost GPU Batch Iterator

## Overview

When training XGBoost on GPU with data too large to fit in GPU memory, use `DeviceQuantileDMatrix` with a custom `DataIter` that streams batches. Each batch is loaded, quantized, then freed — only the quantile sketch stays in memory. Cuts GPU memory usage proportionally to batch count with minimal speed penalty.

## Quick Start

```python
import xgboost as xgb
import numpy as np

class BatchIterator(xgb.core.DataIter):
    def __init__(self, df, features, target, batch_size=256*1024):
        self.df = df
        self.features = features
        self.target = target
        self.batch_size = batch_size
        self.it = 0
        self.n_batches = int(np.ceil(len(df) / batch_size))
        super().__init__()
    
    def reset(self):
        self.it = 0
    
    def next(self, input_data):
        if self.it == self.n_batches:
            return 0
        a = self.it * self.batch_size
        b = min(a + self.batch_size, len(self.df))
        batch = self.df.iloc[a:b]
        input_data(data=batch[self.features], label=batch[self.target])
        self.it += 1
        return 1

# Usage
it = BatchIterator(train_df, features, 'target', batch_size=256*1024)
dtrain = xgb.DeviceQuantileDMatrix(it, max_bin=256)
model = xgb.train(params, dtrain, num_boost_round=2000)
```

## Key Decisions

- **batch_size=256K rows**: balance between memory savings and overhead; tune for your GPU
- **max_bin=256**: matches default; increase to 512 for dense numeric features
- **cuDF acceleration**: pass cuDF DataFrames in `next()` to avoid CPU-GPU transfer
- **Validation set**: create a separate standard DMatrix for eval (usually fits in memory)

## References

- Source: [xgboost-starter-0-793](https://www.kaggle.com/code/cdeotte/xgboost-starter-0-793)
- Competition: American Express - Default Prediction
