---
name: llm-last-token-logit-binary-scoring
description: Score a binary classification prompt by reading the logits of the True/False (or Yes/No) token IDs at the final position and softmaxing only those two values, skipping generation entirely for a 10-50x speedup over decoding
---

## Overview

For binary classification with an instruct LLM, the standard recipe — `generate(max_tokens=1)` and parse the string — is wasteful. The model already computed the full vocabulary distribution at the last input position; you only need two scalar entries from it. Forward-pass once, grab `logits[:, -1, [true_id, false_id]]`, softmax over those two values, and you have `P(True)` as a calibrated probability — not just `0/1`. This is also the *only* way to get a useful continuous score from a model trained with cross-entropy on a single label token, which matters for AUC-style metrics where the threshold can be tuned post-hoc.

## Quick Start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map='auto').eval()

true_id  = tok.encode('True',  add_special_tokens=False)[0]
false_id = tok.encode('False', add_special_tokens=False)[0]

@torch.no_grad()
def score(prompts):
    enc = tok(prompts, return_tensors='pt', padding=True, truncation=True, max_length=2048).to(model.device)
    logits = model(**enc).logits                          # (B, T, V)
    last_idx = enc.attention_mask.sum(1) - 1              # last real token per row
    last = logits[torch.arange(len(prompts)), last_idx]   # (B, V)
    pair = last[:, [true_id, false_id]]                   # (B, 2)
    return torch.softmax(pair.float(), dim=-1)[:, 0]      # P(True)
```

## Quick Start (vLLM variant)

```python
from vllm import SamplingParams
sp = SamplingParams(temperature=0, max_tokens=1, logprobs=20)
out = llm.generate(prompts, sp)
probs = []
for o in out:
    lp = o.outputs[0].logprobs[0]                # dict[token_id -> Logprob]
    pT = lp.get(true_id,  type('x',(),{'logprob':-1e9})()).logprob
    pF = lp.get(false_id, type('x',(),{'logprob':-1e9})()).logprob
    probs.append(np.exp(pT) / (np.exp(pT) + np.exp(pF)))
```

## Workflow

1. Pick a single-token label vocabulary — `True`/`False`, `Yes`/`No`, `A`/`B`. Verify each tokenizes to *exactly one* ID.
2. End your prompt so the next token *is* the answer (e.g. `"... Answer: "` with no trailing space if the tokenizer eats it).
3. Forward-pass with `max_length=2048` (or whatever fits); use left-padding so the last position is meaningful, OR index with `attention_mask.sum(1) - 1` for right-padding.
4. Slice the `[true_id, false_id]` columns from the last-position logits.
5. Softmax over the *pair only* — never over the full vocab — so the two probabilities sum to 1 cleanly.
6. Tune the decision threshold on validation; default 0.5 is rarely optimal for imbalanced binary tasks.

## Key Decisions

- **Softmax over the pair, not full vocab**: full-vocab softmax produces tiny `P(True)` ≈ 1e-3 because mass leaks to thousands of irrelevant tokens; pair-softmax gives a usable probability.
- **Verify single-token labels**: `Yes` is one token in most tokenizers but `Positive` is two — use `tok.encode(label, add_special_tokens=False)` and assert `len == 1`.
- **Mind the leading space**: `' True'` vs `'True'` are different tokens. Match what the prompt's trailing text demands.
- **vs. generation**: generation costs `B * max_tokens * forward_per_token` and discards 99.9% of the logits; this trick costs one forward pass.
- **Use bfloat16 for the forward, float32 for the softmax**: bf16 logits are fine but the softmax of two close values benefits from fp32 precision.
- **Threshold ≠ 0.5**: pick the threshold maximizing your validation metric; on imbalanced data the optimum is often 0.3–0.4.

## References

- [Batch Gemma3 sample rules classification](https://www.kaggle.com/competitions/jigsaw-agile-community-rules)
