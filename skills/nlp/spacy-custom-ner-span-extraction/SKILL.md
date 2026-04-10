---
name: nlp-spacy-custom-ner-span-extraction
description: Train per-class spaCy NER models to extract task-specific spans as custom named entities with compounding batch sizes
domain: nlp
---

# spaCy Custom NER Span Extraction

## Overview

Frame span extraction as NER by creating a custom entity type (e.g. "selected_text") and training separate spaCy NER models per class. Each model specializes in extracting spans relevant to one class (positive, negative). Use compounding batch sizes (start small, grow large) for stable training. Simpler alternative to transformer QA models when data or compute is limited.

## Quick Start

```python
import spacy
from spacy.util import minibatch, compounding
import random

def prepare_ner_data(df, label='selected_text'):
    """Convert DataFrame to spaCy NER training format."""
    train_data = []
    for _, row in df.iterrows():
        text = row['text']
        start = text.find(row['selected_text'])
        if start == -1:
            continue
        end = start + len(row['selected_text'])
        train_data.append((text, {"entities": [(start, end, label)]}))
    return train_data

def train_ner(train_data, n_iter=20, drop=0.5):
    """Train a blank spaCy NER model."""
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner", last=True)
    ner.add_label('selected_text')
    
    nlp.begin_training()
    for _ in range(n_iter):
        random.shuffle(train_data)
        batches = minibatch(train_data, size=compounding(4.0, 500.0, 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, drop=drop)
    return nlp

# Train one model per sentiment class
model_pos = train_ner(prepare_ner_data(df[df.sentiment == 'positive']))
model_neg = train_ner(prepare_ner_data(df[df.sentiment == 'negative']))
```

## Key Decisions

- **Per-class models**: sentiment-specific models outperform one model with sentiment as feature
- **compounding(4, 500, 1.001)**: start with batch=4 for gradient signal, grow to 500 for speed
- **drop=0.5**: aggressive dropout prevents overfitting on small datasets
- **Fallback**: if NER finds no entity, return the full text

## References

- Source: [twitter-sentiment-extaction-analysis-eda-and-model](https://www.kaggle.com/code/tanulsingh077/twitter-sentiment-extaction-analysis-eda-and-model)
- Competition: Tweet Sentiment Extraction
