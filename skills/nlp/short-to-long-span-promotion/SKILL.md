---
name: nlp-short-to-long-span-promotion
description: >
  Promotes a predicted short answer span to its enclosing long-answer candidate by matching token boundaries against pre-extracted document structure.
---
# Short-to-Long Span Promotion

## Overview

In Natural Questions-style QA, models predict both a short answer (exact span) and a long answer (enclosing paragraph/table). Rather than predicting long answers independently, promote the short answer to its enclosing top-level candidate. This leverages the model's more precise short-answer prediction while deriving the long answer from document structure — consistently more accurate than direct long-answer prediction.

## Quick Start

```python
def promote_to_long_answer(short_start, short_end, candidates):
    """Find the smallest top-level candidate enclosing the short span.

    Args:
        short_start: start token index of short answer
        short_end: end token index of short answer
        candidates: list of dicts with start_token, end_token, top_level
    Returns:
        (long_start, long_end) or None if no enclosing candidate
    """
    best = None
    best_len = float("inf")
    for c in candidates:
        if not c.get("top_level", False):
            continue
        if c["start_token"] <= short_start and c["end_token"] >= short_end:
            span_len = c["end_token"] - c["start_token"]
            if span_len < best_len:
                best = (c["start_token"], c["end_token"])
                best_len = span_len
    return best

# Example usage
short_span = (142, 158)  # predicted short answer
long_span = promote_to_long_answer(short_span[0], short_span[1], doc_candidates)
```

## Workflow

1. Predict short answer span using extractive QA model
2. Load pre-extracted document candidates (paragraphs, tables, lists)
3. Find the smallest top-level candidate that fully contains the short span
4. Use that candidate's boundaries as the long answer prediction

## Key Decisions

- **Smallest enclosing**: Pick the tightest enclosing candidate, not the largest
- **Top-level only**: Filter to top-level candidates to avoid nested fragments
- **No short answer**: If short answer is null/unanswerable, long answer should also be null
- **Multiple short answers**: Promote each independently; use the highest-scoring long answer

## References

- [BERT Joint Baseline Notebook](https://www.kaggle.com/code/prokaj/bert-joint-baseline-notebook)
- [BERT Joint](https://www.kaggle.com/code/mmmarchetti/bert-joint)
