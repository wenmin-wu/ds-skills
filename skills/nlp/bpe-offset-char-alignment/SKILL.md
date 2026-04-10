---
name: nlp-bpe-offset-char-alignment
description: Reconstruct character-level offsets for BPE tokens by decoding each token individually and accumulating lengths for precise span mapping
domain: nlp
---

# BPE Offset Character Alignment

## Overview

When a tokenizer doesn't provide reliable character offsets (e.g. some HuggingFace tokenizers with added whitespace), reconstruct them by decoding each BPE token individually and tracking the cumulative character position. Then map character-level span labels to token-level start/end indices by checking which tokens overlap with the annotated span.

## Quick Start

```python
def compute_offsets(token_ids, tokenizer):
    """Reconstruct character offsets by decoding each token."""
    offsets = []
    idx = 0
    for tid in token_ids:
        word = tokenizer.decode([tid])
        offsets.append((idx, idx + len(word)))
        idx += len(word)
    return offsets

def char_span_to_token_span(offsets, char_start, char_end):
    """Map character-level span to token indices."""
    token_start, token_end = None, None
    for i, (a, b) in enumerate(offsets):
        if a < char_end and b > char_start:
            if token_start is None:
                token_start = i
            token_end = i
    return token_start, token_end

# Usage
enc = tokenizer.encode(text, add_special_tokens=False)
offsets = compute_offsets(enc.ids, tokenizer)

# Find where selected_text starts in the original text
char_start = text.find(selected_text)
char_end = char_start + len(selected_text)
tok_start, tok_end = char_span_to_token_span(offsets, char_start, char_end)
```

## Key Decisions

- **Decode-based**: more robust than relying on tokenizer's `.offsets()` which can be buggy
- **Leading spaces**: BPE tokens often include leading spaces — decoded length accounts for this
- **Overlap check**: a token belongs to the span if any part of it overlaps the character range
- **Prefix offset**: add prefix token count (CLS, SEP, query) when mapping to full input positions

## References

- Source: [tensorflow-roberta-0-705](https://www.kaggle.com/code/cdeotte/tensorflow-roberta-0-705)
- Competition: Tweet Sentiment Extraction
