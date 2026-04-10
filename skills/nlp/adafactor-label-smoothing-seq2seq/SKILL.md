---
name: nlp-adafactor-label-smoothing-seq2seq
description: Use Adafactor optimizer with label smoothing for seq2seq fine-tuning — memory-efficient and regularizes overconfident predictions
domain: nlp
---

# Adafactor + Label Smoothing for Seq2Seq

## Overview

Adafactor uses ~3x less memory than Adam by factorizing second-moment estimates. Combined with label smoothing (0.1–0.2), it regularizes the model against overconfident token predictions. Essential for fine-tuning large seq2seq models on limited GPU memory.

## Quick Start

```python
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

args = Seq2SeqTrainingArguments(
    output_dir="./output",
    optim="adafactor",
    label_smoothing_factor=0.15,
    learning_rate=3e-4,
    fp16=False,  # use FP32 for byte-level models (ByT5) to avoid NaN
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # effective batch = 16
    num_train_epochs=10,
    weight_decay=0.01,
    predict_with_generate=True,
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="geo_mean",
)

trainer = Seq2SeqTrainer(model=model, args=args, tokenizer=tokenizer,
    train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=compute_metrics)
```

## Key Decisions

- **Adafactor over Adam**: ~3x memory savings, critical for large models on consumer GPUs
- **label_smoothing=0.15**: prevents overconfident predictions, improves BLEU by 0.5–1.0
- **FP32 for byte-level models**: ByT5 and similar models produce NaN in FP16
- **gradient_accumulation**: simulates larger batch size without extra memory

## References

- Source: [dpc-starter-train](https://www.kaggle.com/code/takamichitoda/dpc-starter-train)
- Competition: Deep Past Challenge - Translate Akkadian to English
