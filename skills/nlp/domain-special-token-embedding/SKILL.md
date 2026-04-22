---
name: nlp-domain-special-token-embedding
description: Add domain-specific categorical values as new special tokens, resize embeddings, and prepend them to input so the model learns domain-aware representations
---

# Domain Special Token Embedding

## Overview

When a categorical feature (patent section, medical specialty, product category) strongly conditions the semantics of the text, encode it as a new special token rather than a text prefix. The model learns a dedicated embedding vector for each category that participates in self-attention. More parameter-efficient and expressive than prepending the category name as text.

## Quick Start

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')

categories = ['[A]', '[B]', '[C]', '[D]', '[E]', '[F]', '[G]', '[H]']
tokenizer.add_special_tokens({'additional_special_tokens': categories})

model = AutoModelForSequenceClassification.from_pretrained(
    'microsoft/deberta-v3-base', num_labels=1)
model.resize_token_embeddings(len(tokenizer))

df['input'] = df['category_token'] + ' ' + df['text_field_1'] + ' [SEP] ' + df['text_field_2']
```

## Workflow

1. Create bracket-wrapped tokens from unique category values: `[A]`, `[B]`, etc.
2. Add as `additional_special_tokens` to the tokenizer
3. Resize model embeddings with `model.resize_token_embeddings(len(tokenizer))`
4. Prepend the category token to each input text
5. New embeddings are randomly initialized — they learn during fine-tuning

## Key Decisions

- **Bracket format**: use `[X]` to avoid collision with existing vocab — tokenizer treats brackets as special
- **Prepend vs append**: prepend so the category attends to all subsequent tokens
- **Number of categories**: works well for < 100 categories; for thousands, use text description instead
- **Frozen vs learned**: let the new embeddings train freely; freeze base embeddings only if overfitting

## References

- [Iterate like a grandmaster!](https://www.kaggle.com/code/jhoward/iterate-like-a-grandmaster)
