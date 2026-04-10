---
name: nlp-bidirectional-translation-augmentation
description: Double training data by adding reverse-direction translation pairs with task prefix prompts
domain: nlp
---

# Bidirectional Translation Augmentation

## Overview

For low-resource translation, double the training data by adding reverse-direction pairs (A→B becomes both A→B and B→A). Uses task-prefix prompts so the model learns both directions in a single model. Especially effective when parallel data is scarce.

## Quick Start

```python
import pandas as pd
from datasets import Dataset

def make_bidirectional(df, src_col, tgt_col, src_lang, tgt_lang):
    fwd = df.copy()
    fwd['input_text'] = f"translate {src_lang} to {tgt_lang}: " + fwd[src_col].astype(str)
    fwd['target_text'] = fwd[tgt_col].astype(str)

    bwd = df.copy()
    bwd['input_text'] = f"translate {tgt_lang} to {src_lang}: " + bwd[tgt_col].astype(str)
    bwd['target_text'] = bwd[src_col].astype(str)

    combined = pd.concat([fwd, bwd], ignore_index=True)
    return Dataset.from_pandas(combined.sample(frac=1, random_state=42))
```

## Key Decisions

- **Task prefix required**: without it, the model can't distinguish direction
- **Shuffle after concat**: prevents the model from seeing all forward then all backward
- **Train only**: keep validation unidirectional to measure real performance
- **Works best with T5/mT5**: encoder-decoder models with task prefixes

## References

- Source: [dpc-starter-train](https://www.kaggle.com/code/takamichitoda/dpc-starter-train)
- Competition: Deep Past Challenge - Translate Akkadian to English
