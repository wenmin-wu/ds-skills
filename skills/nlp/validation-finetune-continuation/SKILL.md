---
name: nlp-validation-finetune-continuation
description: >
  After training on the primary split, continues fine-tuning on the validation set to adapt the model to the target distribution before inference.
---
# Validation Fine-tune Continuation

## Overview

In cross-lingual or domain-shift competitions, training data comes from one distribution (e.g., English) and test data from another (e.g., multilingual). After standard training, continue fine-tuning on the validation set (which shares the test distribution) for a few extra epochs. This squeezes out extra performance by exposing the model to the target domain's patterns before submission.

## Quick Start

```python
# Phase 1: train on primary data
EPOCHS = 3
n_steps = len(x_train) // BATCH_SIZE
model.fit(train_dataset, steps_per_epoch=n_steps,
          validation_data=valid_dataset, epochs=EPOCHS)

# Phase 2: continue on validation data (target distribution)
n_steps = len(x_valid) // BATCH_SIZE
model.fit(valid_dataset.repeat(),
          steps_per_epoch=n_steps,
          epochs=EPOCHS)
```

## Workflow

1. Train model on the main training set for N epochs with validation monitoring
2. After convergence, switch the training data to the validation set
3. Use `.repeat()` if the validation set is small (avoids incomplete epochs)
4. Train for additional epochs with a lower or unchanged learning rate
5. Run inference on the test set using the continued model

## Key Decisions

- **When to use**: Only when validation set matches the test distribution better than training set
- **Epochs**: Typically 1-3 extra epochs; more risks overfitting the small validation set
- **Learning rate**: Keep the same or reduce slightly; large LR can destroy learned features
- **No early stopping**: You lose the held-out monitor, so fix epoch count conservatively
- **Alternative**: Merge train + validation and retrain from scratch (slower but more stable)

## References

- [Jigsaw TPU: XLM-Roberta](https://www.kaggle.com/code/xhlulu/jigsaw-tpu-xlm-roberta)
- [Deep Learning For NLP: Zero To Transformers & BERT](https://www.kaggle.com/code/tanulsingh077/deep-learning-for-nlp-zero-to-transformers-bert)
