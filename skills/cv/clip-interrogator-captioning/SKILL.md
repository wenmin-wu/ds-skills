---
name: cv-clip-interrogator-captioning
description: Generate descriptive text prompts from images by combining BLIP captioning with CLIP cosine similarity against curated label banks for medium, movement, and flavor attributes
---

# CLIP Interrogator Captioning

## Overview

When you need to reverse-engineer or describe an image as a text prompt (e.g., for image-to-prompt tasks), CLIP Interrogator combines two models: BLIP generates a base caption, then CLIP matches the image embedding against precomputed text embeddings from curated label banks (mediums, movements, flavors). The top-matching labels are appended to the caption, producing a rich prompt that captures style, medium, and content.

## Quick Start

```python
import torch
from clip_interrogator import Config, Interrogator

ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
cos = torch.nn.CosineSimilarity(dim=1)

mediums_features = torch.stack([torch.from_numpy(t) for t in ci.mediums.embeds]).to(ci.device)
movements_features = torch.stack([torch.from_numpy(t) for t in ci.movements.embeds]).to(ci.device)
flavors_features = torch.stack([torch.from_numpy(t) for t in ci.flavors.embeds]).to(ci.device)

def interrogate(image):
    caption = ci.generate_caption(image)
    feat = ci.image_to_features(image)
    medium = ci.mediums.labels[cos(feat, mediums_features).topk(1).indices[0]]
    movement = ci.movements.labels[cos(feat, movements_features).topk(1).indices[0]]
    flavors = ", ".join([ci.flavors.labels[i] for i in cos(feat, flavors_features).topk(3).indices])
    return f"{caption}, {medium}, {movement}, {flavors}"
```

## Workflow

1. Load BLIP (captioning) and CLIP (embedding) models via `clip_interrogator`
2. Precompute text embeddings for all label banks (mediums, movements, flavors)
3. For each image: generate BLIP caption, extract CLIP image features
4. Compute cosine similarity against each label bank, take top-k matches
5. Concatenate caption + matched labels into a single prompt string
6. Optionally truncate to fit the CLIP tokenizer's max length (77 tokens)

## Key Decisions

- **Label bank pruning**: removing irrelevant categories (artists, sites) can improve scores by 2-5%
- **Top-k per bank**: 1 for medium/movement, 3 for flavors balances specificity vs. coverage
- **CLIP model**: ViT-L/14 gives best quality; ViT-B/32 is 4x faster with minor quality loss
- **Truncation**: always truncate to tokenizer max length to avoid silent clipping

## References

- [BLIP+CLIP | CLIP Interrogator](https://www.kaggle.com/code/leonidkulyk/lb-0-45836-blip-clip-clip-interrogator)
- [CLIPInterrogator+OFA+ViT](https://www.kaggle.com/code/motono0223/clipinterrogator-ofa-vit)
