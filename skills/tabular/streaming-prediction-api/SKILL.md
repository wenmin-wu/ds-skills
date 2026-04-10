---
name: tabular-streaming-prediction-api
description: >
  Online inference pattern that processes test batches sequentially, updating feature dictionaries incrementally for time-series prediction APIs.
---
# Streaming Prediction API

## Overview

Some competitions (and production systems) serve test data in sequential batches — you predict one batch, then receive the next with updated ground truth. Maintain in-memory feature dictionaries that update after each batch. This pattern handles the Kaggle time-series API and applies to any online learning / streaming prediction scenario.

## Quick Start

```python
from collections import defaultdict
import numpy as np

class StreamingPredictor:
    def __init__(self, model, features, global_mean=0.5):
        self.model = model
        self.features = features
        self.global_mean = global_mean
        self.user_sum = defaultdict(float)
        self.user_count = defaultdict(int)
        self.content_mean = {}  # precomputed from training

    def predict_batch(self, batch_df):
        """Predict and update features for one batch."""
        # Build features from current state
        batch_df['user_mean'] = batch_df['user_id'].map(
            lambda u: self.user_sum[u] / self.user_count[u]
            if self.user_count[u] > 0 else self.global_mean
        )
        batch_df['user_count'] = batch_df['user_id'].map(self.user_count)
        batch_df['content_mean'] = batch_df['content_id'].map(self.content_mean)

        preds = self.model.predict(batch_df[self.features])
        return preds

    def update(self, batch_df, outcomes):
        """Update internal state with observed outcomes."""
        for uid, outcome in zip(batch_df['user_id'], outcomes):
            self.user_sum[uid] += outcome
            self.user_count[uid] += 1

# Usage with Kaggle API
predictor = StreamingPredictor(model, feature_cols)
for test_df, sample_pred_df in env.iter_test():
    preds = predictor.predict_batch(test_df)
    predictor.update(test_df, preds)  # or true outcomes if available
    env.predict(sample_pred_df.assign(answered_correctly=preds))
```

## Workflow

1. Precompute static features from training (content difficulty, etc.)
2. Initialize user-level dictionaries from training history
3. For each test batch: build features → predict → update dictionaries
4. Handle cold-start users with global mean fallback

## Key Decisions

- **Dictionary vs DataFrame**: Dicts are O(1) lookup; DataFrames require merge — dicts are 10x faster for streaming
- **Update timing**: Update AFTER prediction (not before) to avoid leaking current batch
- **Memory**: For millions of users, use `int32` keys and `float32` values
- **Warm start**: Initialize dicts from training data for users seen during training

## References

- Riiid Answer Correctness Prediction (Kaggle)
- Source: [lgbm-with-loop-feature-engineering](https://www.kaggle.com/code/its7171/lgbm-with-loop-feature-engineering)
