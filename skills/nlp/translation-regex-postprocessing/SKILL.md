---
name: nlp-translation-regex-postprocessing
description: Multi-rule regex pipeline to clean seq2seq translation outputs — deduplicate phrases, fix punctuation, remove artifacts
domain: nlp
---

# Translation Regex Post-Processing

## Overview

Seq2seq models produce artifacts: repeated phrases, prompt leakage, trailing fragments, inconsistent punctuation. A cascaded regex pipeline fixes these systematically. Apply after decoding, before MBR or final submission.

## Quick Start

```python
import re

RULES = [
    # Remove leaked prompt prefix
    (re.compile(r'(?i)^translate \w+ to \w+:\s*'), ''),
    # Deduplicate repeated phrases (2-4 word spans)
    (re.compile(r'\b(\w+(?:\s+\w+){1,3})\s+\1\b'), r'\1'),
    # Collapse repeated single words
    (re.compile(r'\b(\w+)(\s+\1){2,}\b'), r'\1'),
    # Remove trailing short fragments
    (re.compile(r'\s+\w{1,3}$'), ''),
    # Normalize multiple spaces
    (re.compile(r'\s{2,}'), ' '),
    # Fix space before punctuation
    (re.compile(r'\s+([.,;:!?])'), r'\1'),
]

def postprocess_translation(text):
    text = text.strip()
    for pattern, repl in RULES:
        text = pattern.sub(repl, text)
    if text and text[-1] not in '.!?"':
        text += '.'
    return text.strip()
```

## Key Decisions

- **Order matters**: prompt removal first, dedup second, punctuation last
- **Sentence ending**: force period if missing — most metrics penalize incomplete sentences
- **Conservative dedup**: only exact phrase repeats, not paraphrases

## References

- Source: [hybrid-best-akkadian](https://www.kaggle.com/code/meenalsinha/hybrid-best-akkadian)
- Competition: Deep Past Challenge - Translate Akkadian to English
