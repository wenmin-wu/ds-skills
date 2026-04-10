---
name: nlp-spatial-dropout-embedding
description: Drop entire embedding channels consistently across all timesteps — preserves temporal structure better than element-wise dropout
domain: nlp
---

# Spatial Dropout on Embeddings

## Overview

Standard dropout on word embeddings randomly zeros individual elements, which can fragment the signal within a single embedding dimension. Spatial dropout drops entire channels (feature dimensions) consistently across all timesteps, forcing the model to not rely on any single feature. More effective for sequential models (LSTM, CNN) where temporal coherence matters.

## Quick Start

```python
import torch
import torch.nn as nn

class SpatialDropout(nn.Dropout2d):
    """Drop entire embedding dimensions across all timesteps."""
    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        x = x.unsqueeze(2)              # (B, T, 1, D)
        x = x.permute(0, 3, 2, 1)       # (B, D, 1, T)
        x = super().forward(x)           # drops entire D channels
        x = x.permute(0, 3, 2, 1)       # (B, T, 1, D)
        x = x.squeeze(2)                # (B, T, D)
        return x

# Usage in model
class TextModel(nn.Module):
    def __init__(self, embedding_matrix, p_drop=0.2):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32))
        self.spatial_dropout = SpatialDropout(p_drop)
        self.lstm = nn.LSTM(300, 128, bidirectional=True, batch_first=True)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.spatial_dropout(x)
        x, _ = self.lstm(x)
        return x
```

## Key Decisions

- **p=0.2**: typical range 0.1–0.3; higher for larger embeddings
- **Before LSTM/CNN**: apply after embedding, before sequential layer
- **Leverages Dropout2d**: reshaping trick reuses PyTorch's spatial dropout implementation
- **Not for transformers**: transformers use attention dropout instead; spatial dropout is for RNN/CNN pipelines

## References

- Source: [simple-lstm-pytorch-version](https://www.kaggle.com/code/bminixhofer/simple-lstm-pytorch-version)
- Competition: Jigsaw Unintended Bias in Toxicity Classification
