---
name: cv-weighted-embedding-blending
description: Ensemble predictions from heterogeneous vision-language models by blending their output embeddings with fixed scalar weights in embedding space
---

# Weighted Embedding Blending

## Overview

When multiple models produce embeddings for the same input (e.g., CLIP k-NN, BLIP interrogator, fine-tuned ViT), blending their output embeddings with scalar weights is a simple yet effective ensemble. Unlike logit averaging, embedding blending works in the target space directly — each model's contribution is weighted and summed before submission.

## Quick Start

```python
import numpy as np

# Embeddings from three different models (shape: [n_samples, embed_dim])
emb_knn = predict_knn(images)       # CLIP k-NN regression
emb_blip = predict_blip(images)     # BLIP/CLIP interrogator + SentenceTransformer
emb_vit = predict_vit(images)       # Fine-tuned ViT regression head

w_knn, w_blip, w_vit = 0.60, 0.15, 0.25
blended = w_knn * emb_knn + w_blip * emb_blip + w_vit * emb_vit
```

## Workflow

1. Generate embeddings from each model independently
2. Ensure all embeddings share the same dimensionality and target space
3. L2-normalize each model's output before blending (prevents scale dominance)
4. Apply fixed scalar weights that sum to 1.0
5. Optionally tune weights on a validation set using grid search or Nelder-Mead

## Key Decisions

- **Weight tuning**: start with equal weights, then grid search in 0.05 increments on validation cosine similarity
- **Normalization**: L2-normalize each model's output before blending to equalize scales
- **Number of models**: 2-4 diverse models; beyond 4, diminishing returns and weight search becomes expensive
- **Diversity matters**: models should differ in architecture or approach (k-NN vs. generative captioning vs. fine-tuned regression)
- **vs. stacking**: embedding blending is simpler and avoids overfitting when validation data is small

## References

- [SDIP CLIP kNNRegression +CLIP+ViT](https://www.kaggle.com/code/lhllmlt/sdip-clip-knnregression-clip-vit)
- [CLIPInterrogator+OFA+ViT](https://www.kaggle.com/code/motono0223/clipinterrogator-ofa-vit)
