---
name: timeseries-stateful-chunk-inference
description: >
  Processes long sequences in fixed-size chunks while carrying RNN hidden state across chunks for memory-efficient inference.
---
# Stateful Chunk Inference

## Overview

Long time series (days of sensor data at 5-sec intervals = 100k+ timesteps) don't fit in GPU memory as a single sequence. Split into fixed-size chunks and process sequentially, passing the hidden state from one chunk to the next. This gives the same result as processing the full sequence but with constant memory usage.

## Quick Start

```python
import torch
import numpy as np

def stateful_inference(model, features, chunk_size=4096, device='cuda'):
    """Run inference on long sequence in chunks, preserving hidden state.

    Args:
        model: RNN/GRU model that returns (predictions, hidden_state)
        features: numpy array of shape (seq_len, n_features)
        chunk_size: max timesteps per forward pass
    """
    seq_len = len(features)
    predictions = np.zeros((seq_len, model.n_classes))
    hidden = None

    model.eval()
    with torch.no_grad():
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            chunk = torch.tensor(features[start:end]).float().unsqueeze(0).to(device)

            preds, hidden = model(chunk, hidden)
            # Detach hidden state to prevent backprop across chunks
            hidden = [h.detach() for h in hidden]

            predictions[start:end] = preds.squeeze(0).cpu().numpy()

    return predictions
```

## Workflow

1. Split input sequence into chunks of fixed size (e.g., 4096 timesteps)
2. Process first chunk with hidden=None (zero-initialized)
3. Pass hidden state from each chunk to the next
4. Detach hidden state tensors to prevent memory accumulation
5. Concatenate chunk predictions into full-length output

## Key Decisions

- **Chunk size**: 4096-8192 is typical; larger = more context per step but more memory
- **Detach hidden**: Essential — without detaching, PyTorch builds a computation graph spanning all chunks
- **Overlap**: No overlap needed for RNNs (hidden state carries context); for CNNs, overlap by receptive field size
- **Batch of series**: Process multiple series in parallel if they fit in memory

## References

- Child Mind Institute - Detect Sleep States (Kaggle)
- Source: [sleep-critical-point-infer](https://www.kaggle.com/code/werus23/sleep-critical-point-infer)
