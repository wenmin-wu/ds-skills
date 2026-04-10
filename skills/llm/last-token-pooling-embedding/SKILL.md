---
name: llm-last-token-pooling-embedding
description: >
  Extracts dense sentence embeddings from decoder-only LLMs by pooling the last non-padding token's hidden state.
---
# Last-Token Pooling Embedding

## Overview

Decoder-only LLMs (Qwen, LLaMA, Mistral) can produce high-quality dense embeddings for retrieval. Unlike encoder models that use [CLS] or mean pooling, causal LMs encode all context into the final token. Extract the last non-padding token's hidden state as the sentence embedding, then L2-normalize for cosine similarity search.

## Quick Start

```python
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

def last_token_pool(hidden_states, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return hidden_states[:, -1]
    seq_lengths = attention_mask.sum(dim=1) - 1
    return hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), seq_lengths]

model = AutoModel.from_pretrained("Qwen/Qwen2.5-14B", torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B")

inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
embeddings = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
embeddings = F.normalize(embeddings, p=2, dim=1)
```

## Workflow

1. Tokenize with left-padding or right-padding (function handles both)
2. Forward pass through the decoder LLM
3. Extract hidden state at the last non-padding position per sequence
4. L2-normalize embeddings for cosine similarity
5. Use dot product or cosine for retrieval scoring

## Key Decisions

- **Left vs right padding**: Left-padding is standard for generation; the function auto-detects
- **Quantization**: Combine with 4-bit NF4 (BitsAndBytes) to fit large models in limited VRAM
- **LoRA fine-tuning**: Train a LoRA adapter with contrastive/triplet loss, then merge for inference
- **Max length**: Longer sequences capture more context but cost quadratic attention

## References

- [EEDI_11_21_14B](https://www.kaggle.com/code/anhvth226/eedi-11-21-14b)
- [Qwen14B_Retrieval_Qwen32B_logits-processor-zoo](https://www.kaggle.com/code/jagatkiran/qwen14b-retrieval-qwen32b-logits-processor-zoo)
- [Eedi Qwen32B vllm with logits-processor-zoo](https://www.kaggle.com/code/aerdem4/eedi-qwen32b-vllm-with-logits-processor-zoo)
