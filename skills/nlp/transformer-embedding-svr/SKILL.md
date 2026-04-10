---
name: nlp-transformer-embedding-svr
description: >
  Extracts frozen embeddings from multiple pretrained transformers and trains SVR on the concatenated features.
---
# Transformer Embedding + SVR

## Overview

Instead of fine-tuning transformers end-to-end, extract mean-pooled embeddings from multiple frozen pretrained models, concatenate them into a high-dimensional feature vector, and train a lightweight SVR on top. This avoids GPU-intensive fine-tuning while achieving competitive scores.

## Quick Start

```python
import torch
from transformers import AutoTokenizer, AutoModel
from cuml.svm import SVR  # GPU-accelerated

def extract_embeddings(texts, model_name, batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().cuda()
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = tokenizer(texts[i:i+batch_size], padding=True,
                         truncation=True, max_length=512, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        embeddings.append(pooled.cpu().numpy())
    return np.vstack(embeddings)

# Concatenate embeddings from multiple models
feats = np.hstack([extract_embeddings(texts, m) for m in model_names])
svr = SVR(C=1)
svr.fit(feats_train, y_train)
```

## Workflow

1. Select 3-5 diverse pretrained models (e.g., DeBERTa-v3-base/large, DeBERTa-v2-xlarge)
2. Extract mean-pooled embeddings from each (frozen, no gradients)
3. Concatenate into single feature matrix (e.g., 5 models x 1024 = 5120 dims)
4. Train one SVR per target variable with GPU-accelerated cuML
5. Average predictions across K folds

## Key Decisions

- **Model diversity**: Mix base/large/xlarge for complementary representations
- **cuML SVR**: 100x faster than sklearn on GPU; critical for 25-fold CV
- **Per-target models**: Train separate SVR per target — each target has different feature importance
- **No fine-tuning**: Trades marginal accuracy for massive speed and simplicity

## References

- Feedback Prize - English Language Learning (Kaggle)
- Source: [rapids-svr-cv-0-450-lb-0-44x](https://www.kaggle.com/code/cdeotte/rapids-svr-cv-0-450-lb-0-44x)
