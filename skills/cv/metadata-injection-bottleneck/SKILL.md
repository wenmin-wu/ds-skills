---
name: cv-metadata-injection-bottleneck
description: Inject scalar metadata (depth, position, clinical features) into U-Net bottleneck via RepeatVector and Reshape for metadata-aware segmentation
---

# Metadata Injection at Bottleneck

## Overview

Segmentation models typically only see pixel data, but scalar metadata (depth, acquisition parameters, patient age) can improve predictions. Inject metadata at the U-Net bottleneck by repeating the feature vector to match spatial dimensions, reshaping, and concatenating with the deepest feature map. This lets the decoder condition on metadata without polluting the encoder's spatial feature extraction.

## Quick Start

```python
from keras.layers import *
from keras.models import Model

input_img = Input((128, 128, 1), name='img')
input_meta = Input((n_features,), name='meta')

# Encoder (produces 8x8 feature map at bottleneck)
# ... encoder layers ...
p4 = MaxPooling2D((2, 2))(c4)  # shape: (8, 8, 256)

# Inject metadata at bottleneck
f = RepeatVector(8 * 8)(input_meta)
f = Reshape((8, 8, n_features))(f)
p4 = concatenate([p4, f], axis=-1)

# Continue with bottleneck + decoder
c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
# ... decoder layers ...

model = Model(inputs=[input_img, input_meta], outputs=[output])
```

## Workflow

1. Define separate Input layers for image and metadata
2. Build encoder as normal (image-only)
3. At the deepest spatial resolution, repeat metadata to match spatial dims
4. Concatenate with bottleneck features
5. Build decoder on the augmented feature map

## Key Decisions

- **Why bottleneck**: metadata is global context — injecting at low resolution avoids interfering with local spatial features
- **RepeatVector + Reshape**: broadcasts N scalar features to (H, W, N) spatial tensor
- **Feature count**: normalize metadata features before injection; raw scale differences can dominate
- **Alternative injection points**: can also inject at skip connections or as FiLM conditioning

## References

- [UNet with depth](https://www.kaggle.com/code/bguberfain/unet-with-depth)
