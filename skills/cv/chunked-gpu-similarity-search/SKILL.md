---
name: cv-chunked-gpu-similarity-search
description: Compute pairwise cosine similarity on GPU in fixed-size chunks to avoid OOM, transferring only threshold-passing results to CPU
domain: cv
---

# Chunked GPU Similarity Search

## Overview

Computing an N×N similarity matrix on GPU runs out of memory for large N (>50K items). Process in chunks: multiply the full embedding matrix by a chunk of rows, threshold on GPU, then transfer only the sparse results to CPU. Reduces peak VRAM from O(N²) to O(N×chunk_size).

## Quick Start

```python
import numpy as np
import torch

def chunked_cosine_matches(embeddings, threshold=0.95, chunk_size=4096):
    """Find matches via chunked cosine similarity on GPU.
    
    Args:
        embeddings: (N, D) L2-normalized numpy array
        threshold: cosine similarity cutoff
        chunk_size: rows per GPU batch
    Returns:
        list of matched index arrays per query
    """
    N = len(embeddings)
    emb_gpu = torch.from_numpy(embeddings).cuda()
    n_chunks = (N + chunk_size - 1) // chunk_size
    
    all_matches = [None] * N
    for j in range(n_chunks):
        a = j * chunk_size
        b = min(a + chunk_size, N)
        sims = torch.matmul(emb_gpu, emb_gpu[a:b].T).T  # (chunk, N)
        sims_cpu = sims.cpu().numpy()
        for k in range(b - a):
            idx = np.where(sims_cpu[k] > threshold)[0]
            all_matches[a + k] = idx
    return all_matches

# Usage: embeddings must be L2-normalized
from sklearn.preprocessing import normalize
embeddings = normalize(raw_embeddings)
matches = chunked_cosine_matches(embeddings, threshold=0.9)
```

## Key Decisions

- **Chunk size 4096**: fits ~4K×N float32 matrix in 16GB VRAM for N≤500K; reduce for larger N
- **L2 normalize first**: makes dot product equal to cosine similarity
- **Threshold on GPU**: `torch.where` before `.cpu()` saves transfer time for sparse results
- **Symmetric optimization**: only compute upper triangle if memory is very tight

## References

- Source: [unsupervised-image-text-baseline-in-20min](https://www.kaggle.com/code/finlay/unsupervised-image-text-baseline-in-20min)
- Competition: Shopee - Price Match Guarantee
