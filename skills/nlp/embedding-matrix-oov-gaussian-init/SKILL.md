---
name: nlp-embedding-matrix-oov-gaussian-init
description: Initialize out-of-vocabulary word embeddings with Gaussian noise matching the pretrained embedding distribution
---

## Overview

When building an embedding matrix from pretrained vectors, words not found in the pretrained vocab default to zero. This pulls OOV tokens to the origin of embedding space — very different from real tokens — and biases downstream models. Instead, sample OOV rows from a Gaussian with the same mean and std as the pretrained embeddings. OOV words then start in the same vector space and the model can learn to distinguish them naturally.

## Quick Start

```python
import numpy as np

def build_embedding_matrix(word_index, embeddings_index, max_features, embed_size):
    # Compute mean/std over pretrained vectors
    all_embs = np.stack(list(embeddings_index.values()))
    emb_mean, emb_std = all_embs.mean(), all_embs.std()

    nb_words = min(max_features, len(word_index))
    # Pre-fill with in-distribution noise
    embedding_matrix = np.random.normal(emb_mean, emb_std,
                                        (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        vector = embeddings_index.get(word)
        if vector is not None:
            embedding_matrix[i] = vector
    return embedding_matrix
```

## Workflow

1. Stack all pretrained embedding vectors to get `emb_mean` and `emb_std`
2. Initialize the full embedding matrix as `N(emb_mean, emb_std)` noise
3. For each word in your vocab, if it exists in the pretrained index, overwrite its row
4. Rows left untouched are OOV but live in the same distribution as real tokens
5. Pass to `nn.Embedding` / Keras `Embedding` with `weights=[matrix]`

## Key Decisions

- **Per-dimension vs scalar stats**: Scalar mean/std is simpler and usually sufficient. Per-dimension stats are a minor refinement.
- **vs. zero-init**: Zero-init anchors OOV to a fixed corner. Gaussian-init lets the model distinguish OOV tokens from real ones via normal variance.
- **trainable=True**: Pair with trainable embeddings so OOV rows can drift toward their semantic neighborhood during training.

## References

- [A look at different embeddings](https://www.kaggle.com/code/sudalairajkumar/a-look-at-different-embeddings)
