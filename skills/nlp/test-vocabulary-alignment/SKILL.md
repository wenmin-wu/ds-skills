---
name: nlp-test-vocabulary-alignment
description: >
  Fits TF-IDF vectorizer on test set first to extract vocabulary, then retrains on train set using that vocabulary for feature consistency.
---
# Test Vocabulary Alignment

## Overview

When train and test have different text distributions (e.g., different essay prompts, different LLM generators), fitting TF-IDF only on train may miss test-specific n-grams. Fit the vectorizer on test first to discover its vocabulary, then refit on train using only that vocabulary. This ensures every feature in your model exists in both train and test.

## Quick Start

```python
from sklearn.feature_extraction.text import TfidfVectorizer

def aligned_tfidf(train_texts, test_texts, **tfidf_kwargs):
    """Fit vocabulary on test, then vectorize train with that vocab."""
    # Step 1: Discover test vocabulary
    vec_test = TfidfVectorizer(**tfidf_kwargs)
    vec_test.fit(test_texts)
    vocab = vec_test.vocabulary_

    # Step 2: Refit on train using test vocabulary
    vec_train = TfidfVectorizer(vocabulary=vocab, **tfidf_kwargs)
    tf_train = vec_train.fit_transform(train_texts)
    tf_test = vec_train.transform(test_texts)

    return tf_train, tf_test, vec_train

tf_train, tf_test, vectorizer = aligned_tfidf(
    train_texts, test_texts,
    ngram_range=(3, 5), sublinear_tf=True, strip_accents='unicode'
)
```

## Workflow

1. Fit TF-IDF on test texts → extract vocabulary
2. Create new TF-IDF with that vocabulary
3. fit_transform on train, transform on test
4. Both matrices now share identical feature columns

## Key Decisions

- **Not leakage**: You use test text (unsupervised) but never test labels — vocabulary discovery is safe
- **When to use**: When test distribution differs from train (different prompts, domains, time periods)
- **Downside**: Misses train-only features that could be useful — combine with train-fit features if needed
- **Alternative**: Fit on train+test combined, then transform separately

## References

- LLM - Detect AI Generated Text (Kaggle)
- Source: [train-your-own-tokenizer](https://www.kaggle.com/code/datafan07/train-your-own-tokenizer)
