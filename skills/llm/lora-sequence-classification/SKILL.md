---
name: llm-lora-sequence-classification
description: Load a pretrained LLM with LoRA adapter via PEFT for memory-efficient fine-tuned sequence classification
domain: llm
---

# LoRA Sequence Classification

## Overview

Fine-tuning a full LLM (7B+ params) for classification is expensive. LoRA (Low-Rank Adaptation) freezes the base model and trains small rank-decomposed weight matrices (~0.1% of params). Load the base model for sequence classification, then apply a trained LoRA adapter via PEFT. Enables 7-9B model inference on a single GPU with fp16/bf16.

## Quick Start

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

# Load base model with classification head
model = AutoModelForSequenceClassification.from_pretrained(
    "google/gemma-2-9b-it",
    num_labels=n_classes,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Apply LoRA adapter
model = PeftModel.from_pretrained(model, "path/to/lora-adapter")
model.eval()

# Inference
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    logits = model(**inputs.to(model.device)).logits
probs = torch.softmax(logits, dim=-1)
```

## Key Decisions

- **bf16/fp16**: halves memory; use bf16 for models trained with it (Gemma, Llama)
- **device_map="auto"**: automatically distributes layers across available GPUs
- **Adapter size**: rank 8-32 typical; higher rank = more capacity but more memory
- **Merge option**: `model.merge_and_unload()` fuses LoRA weights for faster inference

## References

- Source: [gemma2-9b-it-cv-0-945](https://www.kaggle.com/code/cdeotte/gemma2-9b-it-cv-0-945)
- Competition: MAP - Charting Student Math Misunderstandings
