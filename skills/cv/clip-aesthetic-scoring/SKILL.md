---
name: cv-clip-aesthetic-scoring
description: >
  Scores image aesthetic quality by passing L2-normalized CLIP ViT-L/14 embeddings through a trained MLP regressor head.
---
# CLIP Aesthetic Scoring

## Overview

CLIP embeddings capture semantic content but don't directly measure visual quality. The LAION aesthetic predictor adds a small MLP (trained on the SAC+AVA aesthetic datasets) on top of CLIP ViT-L/14 embeddings to predict a 1-10 aesthetic score. This gives a differentiable, batch-friendly quality metric that correlates with human aesthetic judgments. Useful for filtering generated images, ranking candidates, or as a reward signal in RLHF pipelines for image generation.

## Quick Start

```python
import torch
import torch.nn as nn
import clip

class AestheticScorer:
    def __init__(self, clip_model_name='ViT-L/14', mlp_path='aesthetic_mlp.pth'):
        self.clip_model, self.preprocess = clip.load(clip_model_name, device='cuda')
        self.mlp = nn.Sequential(
            nn.Linear(768, 1024), nn.Dropout(0.2),
            nn.Linear(1024, 128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        ).to('cuda')
        self.mlp.load_state_dict(torch.load(mlp_path))
        self.mlp.eval()

    @torch.no_grad()
    def score(self, image):
        img_tensor = self.preprocess(image).unsqueeze(0).to('cuda')
        features = self.clip_model.encode_image(img_tensor)
        features /= features.norm(dim=-1, keepdim=True)
        return self.mlp(features.float()).item() / 10.0

scorer = AestheticScorer()
quality = scorer.score(pil_image)  # 0.0-1.0
```

## Workflow

1. Preprocess image with CLIP's transform
2. Extract CLIP image embeddings
3. L2-normalize the embedding vector
4. Pass through trained MLP regressor
5. Scale output to 0-1 range

## Key Decisions

- **CLIP backbone**: ViT-L/14 is standard; ViT-B/32 is faster but less discriminative
- **MLP weights**: Use LAION's pretrained `sac+logos+ava1-l14-linearMSE.pth` or fine-tune on your domain
- **Threshold**: Scores >0.6 are generally "good"; >0.7 is "high quality"
- **Batching**: Process multiple images in one forward pass for efficiency

## References

- [SD Boost via Default SVG](https://www.kaggle.com/code/taikimori/old-metric-lb-0-694-sd-boost-via-my-default-svg)
