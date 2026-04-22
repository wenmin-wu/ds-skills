---
name: cv-sentence-transformer-target-encoding
description: Encode text prompts into fixed-length dense vectors using SentenceTransformer for cosine-similarity evaluation in image-to-text retrieval tasks
---

# SentenceTransformer Target Encoding

## Overview

In image-to-prompt competitions, the target is not raw text but its SentenceTransformer embedding. Models predict embedding vectors and are scored by cosine similarity against the ground-truth prompt embeddings. Understanding this encoding step is essential: predictions must live in the same embedding space as the evaluation targets.

## Quick Start

```python
from sentence_transformers import SentenceTransformer
import numpy as np

st_model = SentenceTransformer("all-MiniLM-L6-v2")

prompts = ["a painting of a sunset over mountains", "cyberpunk city at night"]
embeddings = st_model.encode(prompts, normalize_embeddings=True)
# embeddings.shape: (2, 384)

# Cosine similarity between two prompt embeddings
similarity = np.dot(embeddings[0], embeddings[1])
```

## Workflow

1. Load a SentenceTransformer model matching the competition's evaluation model
2. Encode all ground-truth prompts into dense vectors (the target)
3. Any prediction pipeline must output vectors in this same space
4. Score predictions via cosine similarity: `np.dot(pred, target) / (norm(pred) * norm(target))`
5. For submission, flatten the embedding matrix row-wise into a single column

## Key Decisions

- **Model choice**: must match evaluation exactly — `all-MiniLM-L6-v2` (384-dim) is common; check competition description
- **Normalization**: enable `normalize_embeddings=True` to simplify cosine similarity to dot product
- **Batch encoding**: use `batch_size=32` for large datasets to avoid OOM
- **Style keywords**: appending style tokens ("fine details, masterpiece") to captions before encoding can bias embeddings toward the target distribution
- **vs. CLIP text embeddings**: SentenceTransformer captures semantic meaning; CLIP captures visual-textual alignment — different spaces

## References

- [BLIP+CLIP | CLIP Interrogator](https://www.kaggle.com/code/leonidkulyk/lb-0-45836-blip-clip-clip-interrogator)
- [Calculating Stable Diffusion Prompt Embeddings](https://www.kaggle.com/code/inversion/calculating-stable-diffusion-prompt-embeddings)
