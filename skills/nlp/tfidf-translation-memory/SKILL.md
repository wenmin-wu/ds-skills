---
name: nlp-tfidf-translation-memory
description: TF-IDF similarity retrieval from a translation memory with SequenceMatcher reranking as a fallback or ensemble component
domain: nlp
---

# TF-IDF Translation Memory

## Overview

For repetitive or formulaic text, retrieve the closest source from a translation memory using TF-IDF similarity, then return its paired target. Combines character n-gram and word n-gram TF-IDF with SequenceMatcher reranking. Works as a standalone system or ensemble component alongside neural MT.

## Quick Start

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import SequenceMatcher
import numpy as np

class TranslationMemory:
    def __init__(self, sources, targets):
        self.targets = targets
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6))
        self.word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
        self.Xc = self.char_vec.fit_transform(sources)
        self.Xw = self.word_vec.fit_transform(sources)
        self.sources = sources

    def retrieve(self, query, top_k=5, min_score=0.3):
        sc = (self.char_vec.transform([query]) @ self.Xc.T).toarray()[0]
        sw = (self.word_vec.transform([query]) @ self.Xw.T).toarray()[0]
        combined = 0.6 * sc + 0.4 * sw
        top_idx = np.argsort(-combined)[:top_k]
        # Rerank with SequenceMatcher
        best_i, best_s = top_idx[0], -1
        for idx in top_idx:
            s = SequenceMatcher(None, query, self.sources[idx]).ratio()
            if s > best_s:
                best_i, best_s = idx, s
        return self.targets[best_i] if combined[best_i] > min_score else None
```

## Key Decisions

- **Char + word TF-IDF**: char n-grams handle typos and morphology, word n-grams capture semantics
- **SequenceMatcher rerank**: edit-distance-based reranking on top candidates improves precision
- **min_score threshold**: below threshold, fall back to neural MT — avoids bad retrievals

## References

- Source: [dpc-starter-infer-add-sentencealign](https://www.kaggle.com/code/qifeihhh666/dpc-starter-infer-add-sentencealign)
- Competition: Deep Past Challenge - Translate Akkadian to English
