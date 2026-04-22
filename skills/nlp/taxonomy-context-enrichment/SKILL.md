---
name: nlp-taxonomy-context-enrichment
description: Enrich model input by mapping categorical codes to human-readable taxonomy descriptions and concatenating them as context for transformer models
---

# Taxonomy Context Enrichment

## Overview

When inputs contain categorical codes (patent CPC codes, ICD medical codes, industry SIC codes), the raw code string carries no semantic information for a language model. Map each code to its human-readable description from an external taxonomy and concatenate it as context. This gives the model domain knowledge without fine-tuning on code semantics.

## Quick Start

```python
import pandas as pd

taxonomy = pd.read_csv('taxonomy_descriptions.csv')  # code → title
df = df.merge(taxonomy, left_on='category_code', right_on='code', how='left')
df['text'] = df['taxonomy_title'] + '[SEP]' + df['anchor'] + '[SEP]' + df['target']

tokenized = tokenizer(df['text'].tolist(), truncation=True,
                       max_length=128, padding=True)
```

## Workflow

1. Obtain the taxonomy lookup table (CSV, API, or parsed from structured files)
2. Map each code in the dataset to its text description via merge/join
3. Concatenate the description as a prefix or context field in the model input
4. Tokenize the enriched text with `[SEP]` delimiters between fields

## Key Decisions

- **Hierarchical codes**: for nested taxonomies (A → A01 → A01B), concatenate parent + child descriptions for richer context
- **Placement**: prepend context before the main input — models attend more to earlier tokens
- **Missing codes**: fill with a generic fallback like "Unknown category" rather than empty string
- **Max length**: taxonomy descriptions add tokens — adjust `max_length` or use the adaptive max-length pattern

## References

- [PPPM / Deberta-v3-large baseline w/ W&B [train]](https://www.kaggle.com/code/yasufuminakama/pppm-deberta-v3-large-baseline-w-w-b-train)
- [[USPPPM] BERT for Patents Baseline [inference]](https://www.kaggle.com/code/ksork6s4/uspppm-bert-for-patents-baseline-inference)
