---
name: llm-two-pass-retrieval-refinement
description: >
  Refines retrieval by running two passes: initial embedding retrieval to get candidates, then LLM-generated text concatenated with the query for a second retrieval pass.
---
# Two-Pass Retrieval Refinement

## Overview

Single-pass dense retrieval can miss when query and target use different vocabulary. Fix this with two passes: (1) retrieve top-K candidates using an initial embedding, (2) feed candidates into an LLM to generate a refined description, (3) concatenate the LLM output with the original query and re-embed for a second retrieval pass. The LLM bridges the vocabulary gap between query and target.

## Quick Start

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

# Pass 1: initial retrieval with sparse query features
q_embeds = model.encode(df["ConstructName"] + ". " + df["SubjectName"])
doc_embeds = model.encode(documents)
top100 = util.semantic_search(q_embeds, doc_embeds, top_k=100)

# Feed top candidates to LLM for refined description
llm_output = llm.generate(build_prompts(df, top100))  # LLM describes best match

# Pass 2: re-retrieve with enriched query
enriched = [f"{llm_out}\n\n{orig}" for llm_out, orig in zip(llm_output, df["full_text"])]
q_embeds_v2 = model.encode(enriched)
top25 = util.semantic_search(q_embeds_v2, doc_embeds, top_k=25)
```

## Workflow

1. Encode queries and documents with a dense embedding model
2. Retrieve top-K1 candidates (K1=50-100) via cosine similarity
3. Format top candidates into an LLM prompt asking for the best match description
4. Concatenate LLM output with original query context
5. Re-encode the enriched query and retrieve top-K2 (K2=25)

## Key Decisions

- **K1 (first pass)**: Large (50-100) to ensure recall; precision doesn't matter yet
- **K2 (second pass)**: Smaller (25) for final ranking
- **LLM output cleaning**: Strip serial numbers or formatting artifacts before re-embedding
- **Embedding model**: Can use same model for both passes, or upgrade for pass 2

## References

- [Eedi Qwen-2.5 32B AWQ two-time retrieval](https://www.kaggle.com/code/takanashihumbert/eedi-qwen-2-5-32b-awq-two-time-retrieval)
