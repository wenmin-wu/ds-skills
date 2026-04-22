---
name: llm-batched-perplexity-scoring
description: Batch-compute perplexity for multiple texts using a causal LM with proper padding, shifted labels, and pad-token masking for efficient GPU utilization
---

# Batched Perplexity Scoring

## Overview

Single-sequence perplexity computation underutilizes the GPU. Batch multiple texts together with left-padding, mask out pad tokens in the loss, and compute perplexity per sequence in one forward pass. This is 4-8x faster than sequential scoring and essential when evaluating thousands of candidate orderings or generations.

## Quick Start

```python
import torch
import transformers
from math import exp

PAD_LABEL = -100

class BatchPerplexityScorer:
    def __init__(self, model_path, device="cuda"):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path, device_map=device, torch_dtype=torch.float16)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    @torch.no_grad()
    def score(self, texts, batch_size=8):
        all_ppl = []
        for i in range(0, len(texts), batch_size):
            batch = [f"{self.tokenizer.bos_token}{t}{self.tokenizer.eos_token}"
                     for t in texts[i:i+batch_size]]
            inputs = self.tokenizer(batch, return_tensors='pt',
                                    padding=True, add_special_tokens=False)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            logits = self.model(**inputs).logits
            labels = inputs['input_ids'].clone()
            labels[labels == self.tokenizer.pad_token_id] = PAD_LABEL
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = self.loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)).view(len(batch), -1)
            valid = (shift_labels != PAD_LABEL).sum(dim=-1)
            ppl = [exp((loss[j].sum() / valid[j]).item()) for j in range(len(batch))]
            all_ppl.extend(ppl)
        return all_ppl
```

## Workflow

1. Set tokenizer to left-padding with `pad_token = eos_token`
2. Wrap each text with BOS/EOS tokens, tokenize as batch with padding
3. Forward pass through causal LM to get logits
4. Replace pad token IDs with -100 in labels (ignored by loss)
5. Compute per-token cross-entropy, sum per sequence, divide by valid length
6. Exponentiate to get perplexity

## Key Decisions

- **Left padding**: causal LMs attend left-to-right; left-pad preserves token positions
- **PAD_LABEL = -100**: PyTorch CrossEntropyLoss ignores this index automatically
- **BOS/EOS wrapping**: manual wrapping with `add_special_tokens=False` ensures exact control
- **Quantization**: combine with 4-bit/8-bit for larger models on limited GPU

## References

- [Brute Force First Sample - Perplexity 470](https://www.kaggle.com/code/cdeotte/brute-force-first-sample-perplexity-470)
