---
name: nlp-multi-temperature-candidate-sampling
description: Generate diverse translation candidates by running nucleus sampling at multiple temperatures then pooling for MBR selection
domain: nlp
---

# Multi-Temperature Candidate Sampling

## Overview

A single temperature produces homogeneous candidates. Run nucleus sampling at 3+ temperatures (e.g. 0.55, 0.75, 1.05) to generate candidates with varying diversity levels, then pool all for MBR selection. Combines with beam search candidates for maximum coverage.

## Quick Start

```python
def generate_diverse_candidates(model, tokenizer, input_ids, attn_mask,
                                 temperatures=[0.55, 0.75, 1.05],
                                 samples_per_temp=4, top_p=0.9):
    all_candidates = []
    # Beam search candidates
    beam_out = model.generate(input_ids=input_ids, attention_mask=attn_mask,
        num_beams=8, num_return_sequences=4, early_stopping=True, max_new_tokens=256)
    all_candidates.extend(tokenizer.batch_decode(beam_out, skip_special_tokens=True))
    # Multi-temperature sampling
    for temp in temperatures:
        samp_out = model.generate(input_ids=input_ids, attention_mask=attn_mask,
            do_sample=True, temperature=temp, top_p=top_p,
            num_return_sequences=samples_per_temp, max_new_tokens=256)
        all_candidates.extend(tokenizer.batch_decode(samp_out, skip_special_tokens=True))
    return all_candidates
```

## Key Decisions

- **3 temperatures**: low (0.55) for safe variants, mid (0.75) for balanced, high (1.05) for creative
- **Beam + sampling**: beam gives high-probability candidates, sampling adds diversity
- **top_p=0.9**: nucleus sampling avoids degenerate low-probability tokens
- **Pool size**: ~16-20 candidates per input (4 beam + 4×3 sampling) is a good default

## References

- Source: [lb-35-9-with-regex-corrections-public-model](https://www.kaggle.com/code/vitorhugobarbedo/lb-35-9-with-regex-corrections-public-model)
- Competition: Deep Past Challenge - Translate Akkadian to English
