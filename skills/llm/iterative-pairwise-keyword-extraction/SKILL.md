---
name: llm-iterative-pairwise-keyword-extraction
description: >
  Iteratively prompts an LLM over document pairs to extract and deduplicate keywords, building a comprehensive term set from multiple perspectives.
---
# Iterative Pairwise Keyword Extraction

## Overview

A single LLM prompt over one document produces narrow keywords biased by the prompt framing. By iterating over pairs — (anchor document, neighbor 1), (anchor, neighbor 2), etc. — each prompt highlights different aspects of the anchor. Keywords from all iterations are merged and deduplicated, producing a richer term set than any single extraction. This is especially effective for retrieval tasks where the query needs to cover multiple facets of the source document.

## Quick Start

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

TEMPLATE = """Given these two documents, extract the key technical terms 
that describe what the first document is about.

Document A: {title_a}
{abstract_a}

Document B: {title_b}
{abstract_b}

Keywords (comma separated):"""

def extract_keywords_pairwise(anchor, neighbors, model, tokenizer, max_tokens=50):
    all_keywords = set()
    for neighbor in neighbors:
        prompt = TEMPLATE.format(
            title_a=anchor['title'], abstract_a=anchor['abstract'],
            title_b=neighbor['title'], abstract_b=neighbor['abstract'],
        )
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        output_ids = model.generate(input_ids, max_new_tokens=max_tokens,
                                     do_sample=False)
        answer = tokenizer.decode(output_ids[0][input_ids.shape[1]:],
                                   skip_special_tokens=True)
        # Parse comma/semicolon/newline separated keywords
        for sep in [",", ";", "\n"]:
            if sep in answer:
                terms = answer.split(sep)
                break
        else:
            terms = [answer]
        all_keywords.update(t.strip().lower() for t in terms if len(t.strip()) < 40)
    return list(all_keywords)

# Get 5 nearest neighbors via embedding similarity
neighbors = get_nearest(anchor, corpus, k=5)
keywords = extract_keywords_pairwise(anchor, neighbors, model, tokenizer)
```

## Workflow

1. For each anchor document, retrieve K nearest neighbors (by embedding similarity or BM25)
2. Prompt the LLM with each (anchor, neighbor) pair
3. Parse keywords from each response (handle comma, semicolon, newline delimiters)
4. Merge and deduplicate across all iterations
5. Use the combined keyword set for downstream retrieval or classification

## Key Decisions

- **Number of neighbors**: 3–5 balances coverage vs compute cost
- **Deduplication**: Lowercase + strip; optionally fuzzy-match near-duplicates
- **Length filter**: Drop keywords > 40 chars — likely extraction artifacts
- **Neighbor source**: Embedding similarity finds semantic neighbors; BM25 finds lexical ones — combine both

## References

- [USPTO: KerasNLP Starter](https://www.kaggle.com/code/awsaf49/uspto-kerasnlp-starter)
