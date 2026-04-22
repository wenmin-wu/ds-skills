---
name: llm-sentence-truncation-fallback
description: Truncate LLM output to exactly N sentences and fall back to a known-good baseline string when output is empty or too short
---

# Sentence Truncation with Fallback

## Overview

LLM outputs for structured tasks (prompt recovery, classification rationale, single-line answers) often include unwanted preamble, repetition, or rambling. Truncate to the first N sentences, then check length — if the result is too short or empty, substitute a well-scoring baseline string. This two-step post-processing ensures consistent submission quality.

## Quick Start

```python
import re

BASELINE = "Improve the following text using a more formal and academic tone while maintaining the original meaning"

def truncate_sentences(text, max_sentences=1):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    result = ' '.join(sentences[:max_sentences])
    return result

def postprocess(raw_output, min_length=15):
    cleaned = truncate_sentences(raw_output, max_sentences=1)
    cleaned = cleaned.strip().strip('"').strip("'")
    if len(cleaned) < min_length:
        return BASELINE
    return cleaned

predictions = [postprocess(output) for output in raw_outputs]
```

## Workflow

1. Split raw LLM output by sentence boundaries
2. Keep only the first N sentences (typically 1 for single-prompt tasks)
3. Strip quotes, whitespace, and formatting artifacts
4. Check minimum length — outputs under threshold are likely garbage
5. Replace short/empty outputs with a high-scoring baseline string

## Key Decisions

- **Baseline string**: use a generic, broadly applicable prompt that scores well on average (~0.60+ similarity)
- **min_length**: 15-20 characters filters out single-word or empty outputs
- **Sentence count**: 1 for prompt recovery; 2-3 for summaries or explanations
- **Combine with timeout**: apply fallback also when inference exceeds time budget

## References

- [Mistral 7B Prompt Recovery Version 2](https://www.kaggle.com/code/richolson/mistral-7b-prompt-recovery-version-2)
