---
name: nlp-sentiment-conditioned-qa-span
description: Prepend a sentiment token as the query in a QA-style input to condition span extraction on sentiment class without architectural changes
domain: nlp
---

# Sentiment-Conditioned QA Span Extraction

## Overview

For extracting sentiment-bearing spans from text, frame it as a QA task where the "question" is a single sentiment token. Encode the sentiment label (positive/negative/neutral) as a specific vocabulary token ID and prepend it before the text. The model learns to extract different spans depending on which sentiment token is present. No architecture changes needed — just input formatting.

## Quick Start

```python
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Map sentiment labels to specific token IDs
sentiment_id = {
    'positive': tokenizer.encode('positive', add_special_tokens=False)[0],
    'negative': tokenizer.encode('negative', add_special_tokens=False)[0],
    'neutral': tokenizer.encode('neutral', add_special_tokens=False)[0],
}

def encode_qa_input(text, sentiment, tokenizer, max_len=96):
    """Build QA-style input: [CLS] sentiment_token [SEP] [SEP] text [SEP]"""
    enc = tokenizer.encode(text, add_special_tokens=False)
    s_tok = sentiment_id[sentiment]
    
    input_ids = [tokenizer.cls_token_id] + [s_tok] + \
                [tokenizer.sep_token_id] * 2 + enc + [tokenizer.sep_token_id]
    token_type_ids = [0] * 4 + [0] * (len(enc) + 1)
    attention_mask = [1] * len(input_ids)
    
    # Pad
    pad_len = max_len - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * pad_len
    token_type_ids += [0] * pad_len
    attention_mask += [0] * pad_len
    
    return input_ids, attention_mask, token_type_ids
```

## Key Decisions

- **Single token query**: one sentiment token is enough — full-word queries add noise
- **Token ID lookup**: use the tokenizer's own vocab ID for the sentiment word
- **Offset shift**: span targets must shift by the number of prefix tokens (typically +4)
- **Works with any transformer**: RoBERTa, BERT, DeBERTa — just adjust special token IDs

## References

- Source: [roberta-inference-5-folds](https://www.kaggle.com/code/abhishek/roberta-inference-5-folds)
- Competition: Tweet Sentiment Extraction
