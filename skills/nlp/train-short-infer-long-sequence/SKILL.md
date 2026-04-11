---
name: nlp-train-short-infer-long-sequence
description: >
  Trains a transformer at shorter sequence length for speed, then runs inference at a longer sequence length to capture more context, exploiting position embedding generalization.
---
# Train Short, Infer Long Sequence

## Overview

Transformer training cost scales quadratically with sequence length. Training at 1024 tokens is 4x faster than 2048. But at inference time, longer contexts improve predictions — especially for NER where entity meaning depends on surrounding sentences. This technique trains at a short sequence length (e.g., 1024) then infers at a longer one (e.g., 2048). Modern transformers with relative position embeddings (DeBERTa, RoPE-based models) generalize well to unseen lengths. The result: training-time savings with inference-time accuracy.

## Quick Start

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

TRAIN_MAX_LEN = 1024   # shorter for fast training
INFER_MAX_LEN = 2048   # longer for better context at inference

# Training: tokenize at short length
train_encodings = tokenizer(
    train_texts, max_length=TRAIN_MAX_LEN,
    truncation=True, padding='max_length',
    return_offsets_mapping=True
)

# Inference: tokenize at longer length
test_encodings = tokenizer(
    test_texts, max_length=INFER_MAX_LEN,
    truncation=True, padding='max_length',
    return_offsets_mapping=True
)

# Same model handles both — no retraining needed
```

## Workflow

1. Set a short `max_length` for training tokenization (e.g., 1024)
2. Train the model normally at this length
3. At inference, set a longer `max_length` (e.g., 2048)
4. The model generalizes to the longer context without retraining
5. Use sliding window if texts exceed even the longer inference length

## Key Decisions

- **Model compatibility**: Works best with relative position encodings (DeBERTa, ALiBi, RoPE); absolute position models (BERT) degrade beyond training length
- **Length ratio**: 2x is safe (1024→2048); 4x may degrade for some models
- **Memory at inference**: Longer sequences need more GPU memory — reduce batch size at inference
- **Sliding window**: For texts exceeding inference length, use overlapping windows and merge predictions

## References

- [PII Data Detection: KerasNLP Starter](https://www.kaggle.com/code/awsaf49/pii-data-detection-kerasnlp-starter-notebook)
