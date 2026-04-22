---
name: tabular-multi-input-embedding-nn
description: Keras multi-input model with separate embedding layers for categoricals, GRU for text sequences, and dense layers for numerics, all concatenated into a shared regression trunk
---

# Multi-Input Embedding Neural Network

## Overview

For datasets mixing categorical, text, and numeric features, build a single Keras model with dedicated input branches: Embedding layers for each categorical, Embedding+GRU for text sequences, and a passthrough for numerics. Concatenate all branches into shared dense layers. This avoids lossy manual encoding and lets the network learn feature interactions end-to-end.

## Quick Start

```python
from tensorflow.keras.layers import (Input, Embedding, GRU, Dense,
                                     Flatten, Dropout, concatenate)
from tensorflow.keras.models import Model

name = Input(shape=(MAX_NAME_LEN,))
desc = Input(shape=(MAX_DESC_LEN,))
brand = Input(shape=(1,))
condition = Input(shape=(1,))
num_vars = Input(shape=(NUM_FEATURES,))

emb_desc = Embedding(VOCAB_SIZE, 60)(desc)
emb_name = Embedding(VOCAB_SIZE, 20)(name)
emb_brand = Embedding(NUM_BRANDS, 10)(brand)
emb_cond = Embedding(NUM_CONDITIONS, 5)(condition)

gru_desc = GRU(16)(emb_desc)
gru_name = GRU(8)(emb_name)

x = concatenate([gru_desc, gru_name, Flatten()(emb_brand),
                 Flatten()(emb_cond), num_vars])
x = Dropout(0.1)(Dense(128, activation='relu')(x))
x = Dropout(0.1)(Dense(64, activation='relu')(x))
output = Dense(1, activation='linear')(x)

model = Model([name, desc, brand, condition, num_vars], output)
model.compile(optimizer='adam', loss='mse')
```

## Workflow

1. Tokenize text fields with a shared `Tokenizer`, pad to fixed lengths
2. LabelEncode categorical fields on train+test combined
3. Define one `Input` per feature group
4. Route each input through its branch (Embedding, Embedding+GRU, or passthrough)
5. Concatenate all branch outputs and feed into dense layers
6. Train with appropriate loss (MSE for regression, BCE for classification)

## Key Decisions

- **Shared vs separate tokenizer**: shared vocabulary across text fields reduces vocab size
- **GRU vs LSTM**: GRU is faster with comparable performance for short texts
- **Embedding dim**: 5-10 for low-cardinality categoricals, 20-60 for text
- **Dropout placement**: after each dense layer, not inside GRU for short sequences

## References

- [A simple nn solution with Keras](https://www.kaggle.com/code/knowledgegrappler/a-simple-nn-solution-with-keras-0-48611-pl)
