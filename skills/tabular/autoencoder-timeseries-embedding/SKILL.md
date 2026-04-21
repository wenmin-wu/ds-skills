---
name: tabular-autoencoder-timeseries-embedding
description: Train a PyTorch autoencoder on time-series summary statistics to produce dense encoded features for downstream GBDT models
---

# Autoencoder Time-Series Embedding

## Overview

When per-subject time-series data is too large or variable-length to feed directly into tabular models, compress the `.describe()` statistics through an autoencoder. The encoder's bottleneck produces a fixed-size dense embedding that captures non-linear relationships between summary stats. Works well when raw stats are high-dimensional (50-200 columns) and correlated.

## Quick Start

```python
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, enc_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, enc_dim * 3), nn.ReLU(),
            nn.Linear(enc_dim * 3, enc_dim * 2), nn.ReLU(),
            nn.Linear(enc_dim * 2, enc_dim), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(enc_dim, input_dim * 2), nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim * 3), nn.ReLU(),
            nn.Linear(input_dim * 3, input_dim), nn.Sigmoid())

    def forward(self, x):
        return self.decoder(self.encoder(x))

def encode_features(df_stats, enc_dim=50, epochs=50, bs=32):
    scaler = StandardScaler()
    X = torch.FloatTensor(scaler.fit_transform(df_stats.fillna(0)))
    ae = AutoEncoder(X.shape[1], enc_dim)
    opt = torch.optim.Adam(ae.parameters())
    for _ in range(epochs):
        for i in range(0, len(X), bs):
            batch = X[i:i+bs]
            loss = nn.MSELoss()(ae(batch), batch)
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        enc = ae.encoder(X).numpy()
    return pd.DataFrame(enc, columns=[f'enc_{i}' for i in range(enc_dim)])
```

## Workflow

1. Compute `df.describe()` on each subject's time-series → flatten into one row per subject
2. Fill NaN with 0, then `StandardScaler` to normalize
3. Train autoencoder (MSE loss) on the full dataset (train + test) — unsupervised, no leakage
4. Extract encoder output as new features
5. Merge encoded features with main tabular DataFrame for GBDT training

## Key Decisions

- **Encoding dim**: 50 is a good default for 100-200 input stats; reduce to 20-30 for smaller inputs
- **Include test data**: yes — autoencoder is unsupervised, using test improves the learned manifold
- **Epochs**: 50 is usually sufficient; monitor reconstruction loss for convergence
- **Alternative**: PCA/SVD is faster but misses non-linear structure

## References

- [LB0.494 with TabNet](https://www.kaggle.com/code/ichigoe/lb0-494-with-tabnet)
- [CMI | Tuning | Ensemble of solutions](https://www.kaggle.com/code/batprem/cmi-tuning-ensemble-of-solutions)
