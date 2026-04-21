---
name: nlp-multi-retriever-union-ensemble
description: Run multiple independent retrieve-rerank pipelines and union-merge their predicted IDs per query via explode-groupby-unique
---

# Multi-Retriever Union Ensemble

## Overview

Different retrievers (varying model architectures, embedding sizes, training data) recall different relevant items. Instead of averaging scores, run each pipeline end-to-end independently, then union all predicted IDs per query. This maximizes recall without complex score calibration across models.

## Quick Start

```python
import pandas as pd

configs = [cfg_roberta, cfg_minilm, cfg_mpnet]
submissions = []

for i, cfg in enumerate(configs):
    emb_q, emb_d = encode(queries, docs, cfg.model)
    candidates = knn_retrieve(emb_q, emb_d, top_n=cfg.top_n)
    pairs = build_pairs(queries, docs, candidates)
    scores = rerank(pairs, cfg.cross_encoder)
    sub = threshold_and_group(scores, cfg.threshold)
    sub.to_csv(f'sub_{i}.csv', index=False)
    submissions.append(sub)

# Union ensemble
merged = pd.concat(submissions)
merged['content_ids'] = merged['content_ids'].str.split(' ')
merged = (merged.explode('content_ids')
          .groupby('topic_id')['content_ids']
          .unique().reset_index())
merged['content_ids'] = merged['content_ids'].apply(' '.join)
```

## Workflow

1. Define N configs with different bi-encoders and/or cross-encoders
2. Run each retrieve-rerank pipeline independently → N submission DataFrames
3. Concatenate all submissions
4. Explode space-separated IDs into individual rows
5. `groupby(query_id).unique()` to deduplicate
6. Rejoin IDs into space-separated strings

## Key Decisions

- **Model diversity**: vary architecture (RoBERTa vs MiniLM vs MPNet) rather than just random seeds
- **Union vs intersection**: union maximizes recall (good for MAP/F2); intersection maximizes precision
- **Independent thresholds**: each pipeline uses its own optimal threshold before merging
- **Diminishing returns**: 2-3 diverse models capture most gains; beyond 5 adds little recall

## References

- [LECR-stsb_roberta_base](https://www.kaggle.com/code/yuiwai/lecr-stsb-roberta-base)
- [all-minilm-l6-v2 tuning_model_add](https://www.kaggle.com/code/yuiwai/all-minilm-l6-v2-tuning-model-add)
