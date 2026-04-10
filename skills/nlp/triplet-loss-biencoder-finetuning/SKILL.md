---
name: nlp-triplet-loss-biencoder-finetuning
description: >
  Fine-tunes a bi-encoder with triplet loss using retrieval-mined hard negatives for dense similarity search.
---
# Triplet Loss Bi-Encoder Fine-Tuning

## Overview

Off-the-shelf embedding models often underperform on domain-specific retrieval. Fine-tune a bi-encoder (BGE, SentenceTransformer) with triplet loss: anchor (query), positive (correct match), negative (hard negative mined from retrieval). Hard negatives — high-similarity but incorrect items — force the model to learn fine-grained distinctions.

## Quick Start

```python
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.losses import TripletLoss
from sentence_transformers.training_args import BatchSamplers

model = SentenceTransformer("BAAI/bge-large-en-v1.5")
loss = TripletLoss(model)

# Dataset columns: anchor, positive, negative
# Mine hard negatives: encode all docs, retrieve top-K, take non-matching as negatives
args = SentenceTransformerTrainingArguments(
    output_dir="./finetuned-bge",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    lr_scheduler_type="cosine_with_restarts",
)
trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=triplets, loss=loss)
trainer.train()
```

## Workflow

1. Encode all documents with the pretrained model
2. For each query, retrieve top-K nearest documents via cosine similarity
3. Non-matching retrieved items become hard negatives; correct match is the positive
4. Format as (anchor, positive, negative) triplets
5. Fine-tune with `TripletLoss` and `NO_DUPLICATES` batch sampler
6. Evaluate with MAP@K or Recall@K on a held-out set

## Key Decisions

- **Hard negative mining**: Re-mine negatives every few epochs as the model improves
- **Batch sampler**: `NO_DUPLICATES` ensures diverse batches; avoids trivial in-batch negatives
- **Loss function**: TripletLoss with default margin (5.0); alternatives: MultipleNegativesRankingLoss
- **Learning rate**: 1e-5 to 3e-5; too high destabilizes pretrained weights

## References

- [Fine-tuning bge [Train]](https://www.kaggle.com/code/sinchir0/fine-tuning-bge-train)
