---
name: nlp-sentence-alignment-augmentation
description: Split multi-sentence parallel pairs into aligned sentence pairs to expand training data for seq2seq models
domain: nlp
---

# Sentence Alignment Augmentation

## Overview

Many parallel corpora contain document-level pairs where multiple sentences are joined. Split them into sentence-level pairs when source and target sentence counts match. This multiplies training examples and helps the model learn finer-grained alignments.

## Quick Start

```python
import re
import pandas as pd

def sentence_align(df, src_col='source', tgt_col='target'):
    aligned = []
    for _, row in df.iterrows():
        src, tgt = str(row[src_col]), str(row[tgt_col])
        tgt_sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', tgt) if s.strip()]
        src_lines = [s.strip() for s in src.split('\n') if s.strip()]
        if len(tgt_sents) > 1 and len(tgt_sents) == len(src_lines):
            for s, t in zip(src_lines, tgt_sents):
                if len(s) > 3 and len(t) > 3:
                    aligned.append({src_col: s, tgt_col: t})
        else:
            aligned.append({src_col: src, tgt_col: tgt})
    return pd.DataFrame(aligned)
```

## Key Decisions

- **Count match only**: only split when source and target have equal sentence counts — avoids misalignment
- **Min length filter**: skip fragments < 3 chars to avoid noise
- **Keep originals**: if counts don't match, keep the full pair as-is
- **Apply to train only**: never split validation/test data

## References

- Source: [dpc-starter-train](https://www.kaggle.com/code/takamichitoda/dpc-starter-train)
- Competition: Deep Past Challenge - Translate Akkadian to English
