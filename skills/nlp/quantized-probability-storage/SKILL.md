---
name: nlp-quantized-probability-storage
description: >
  Stores model softmax probabilities as uint8 (0-255) to reduce RAM by 4x during multi-model ensemble inference.
---
# Quantized Probability Storage

## Overview

Ensembling multiple large transformer models requires storing all predictions in memory before averaging. With float32 probabilities, 5 models x 10K documents x 4096 tokens x 15 classes = ~12 GB. Quantizing to uint8 (0-255) reduces this to ~3 GB with negligible accuracy loss. Probabilities are multiplied by 255, cast to byte, stored, then divided by 255 when needed for averaging.

## Quick Start

```python
import numpy as np
import torch

def quantize_probs(probs, max_length=4096):
    """Convert float32 probabilities to uint8 for storage.

    Args:
        probs: (batch, seq_len, n_classes) float tensor
    Returns:
        (batch, max_length, n_classes) uint8 numpy array
    """
    quantized = (probs * 255).byte().cpu().numpy()
    # Pad or truncate to fixed length
    if quantized.shape[1] > max_length:
        quantized = quantized[:, :max_length, :]
    elif quantized.shape[1] < max_length:
        pad_width = ((0, 0), (0, max_length - quantized.shape[1]), (0, 0))
        quantized = np.pad(quantized, pad_width, constant_values=0)
    return quantized

def dequantize_and_average(all_preds, n_models):
    """Average quantized predictions from multiple models."""
    result = np.zeros_like(all_preds[0], dtype=np.float32)
    for preds in all_preds:
        result += preds.astype(np.float32) / 255.0
    return result / n_models
```

## Workflow

1. Run each model's inference, get softmax probabilities
2. Quantize: multiply by 255, cast to uint8
3. Pad/truncate to fixed sequence length for uniform storage
4. Store quantized arrays in a list
5. When averaging: cast back to float32, divide by 255, then mean

## Key Decisions

- **Precision loss**: Max error is 1/255 ≈ 0.004 per value — negligible for ensemble averaging
- **Memory savings**: 4x reduction (float32 → uint8); critical for 5+ model ensembles
- **Alternative**: float16 gives 2x savings with zero precision loss, but uint8 is better for 4+ models
- **When to use**: Ensemble inference with memory constraints; not needed for single models

## References

- [Infer Fast Ensemble Models](https://www.kaggle.com/code/librauee/infer-fast-ensemble-models)
