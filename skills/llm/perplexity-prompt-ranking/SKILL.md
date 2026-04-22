---
name: llm-perplexity-prompt-ranking
description: Rank candidate prompts by computing LLM perplexity of the full conversation conditioned on each prompt, selecting the lowest-perplexity candidate as the best match
---

# Perplexity Prompt Ranking

## Overview

When selecting among multiple candidate prompts (e.g., which instruction produced a given output), format each candidate into a full conversation template with the input/output text, compute the perplexity of each formatted sequence, and rank by lowest perplexity. The LLM assigns lower perplexity to sequences that are more "natural" given its training — the prompt that actually produced the output will typically score lowest.

## Quick Start

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

def rank_prompts(candidates, context_template, model, tokenizer):
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    scores = []
    texts = [context_template.format(prompt=p) for p in candidates]
    inputs = tokenizer(texts, return_tensors="pt", padding=True,
                       truncation=True, add_special_tokens=False).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    labels = inputs["input_ids"].clone()
    labels[~inputs["attention_mask"].bool()] = -100
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    for i in range(len(candidates)):
        loss = loss_fn(shift_logits[i], shift_labels[i])
        valid = (shift_labels[i] != -100).sum()
        scores.append((loss.sum() / valid).item())
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1])
    return ranked[0][0], ranked

best_prompt, all_ranked = rank_prompts(
    ["Rewrite formally", "Make it rhyme", "Simplify"],
    "Instruction: {prompt}\nInput: ...\nOutput: ...",
    model, tokenizer)
```

## Workflow

1. Define a set of candidate prompts
2. Format each into a full conversation template with input/output context
3. Batch-tokenize and run a single forward pass
4. Compute per-sequence perplexity (cross-entropy / valid tokens)
5. Select the candidate with the lowest perplexity

## Key Decisions

- **Template matters**: the candidate must appear in the same position as the real prompt would
- **Mask padding**: set pad token labels to -100 so they don't affect the score
- **vs generation**: ranking is faster than generating — one forward pass vs autoregressive decoding
- **Candidate pool**: larger pools find better matches but cost linearly more compute

## References

- [Perplexity Baseline Phi-2 / Gemma-7B-IT](https://www.kaggle.com/code/itahiro/perplexity-baseline-phi-2-gemma-7b-it)
