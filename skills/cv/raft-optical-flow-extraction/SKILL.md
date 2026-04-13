---
name: cv-raft-optical-flow-extraction
description: Extract dense per-pixel motion fields between consecutive video frames using a pretrained RAFT model, producing an HxWx2 flow tensor that can be channel-stacked with RGB or used as a standalone motion feature for action / impact / event detection
---

## Overview

For any task where motion matters more than appearance — collision detection, action recognition, anomaly localization — feeding a CNN raw RGB pairs is wasteful: the network has to rediscover motion from scratch on every frame pair. Pretrained RAFT (Recurrent All-Pairs Field Transforms) gives you dense optical flow as a precomputed input channel, with state-of-the-art accuracy on Sintel/KITTI and a one-call API. The output is an `HxWx2` field where each pixel holds (dx, dy) displacement in pixels. Stack it as channels 4-5 of an RGB+flow input, or convert to magnitude/angle and use as a standalone heatmap.

## Quick Start

```python
import torch, cv2, numpy as np
from raft.core.raft import RAFT
from raft.core.utils.utils import InputPadder
from raft.config import RAFTConfig

config = RAFTConfig(dropout=0, alternate_corr=False, small=False, mixed_precision=False)
model = RAFT(config).to(device).eval()
model.load_state_dict(torch.load('raft-sintel.pth', map_location=device))

def to_tensor(img):
    return torch.from_numpy(img).permute(2, 0, 1).float()[None].to(device)

@torch.no_grad()
def flow_between(f1, f2, iters=20):
    i1, i2 = to_tensor(f1), to_tensor(f2)
    padder  = InputPadder(i1.shape)
    i1, i2  = padder.pad(i1, i2)
    _, flow = model(i1, i2, iters=iters, test_mode=True)
    return flow[0].permute(1, 2, 0).cpu().numpy()      # (H, W, 2)
```

## Workflow

1. Download the RAFT checkpoint matching your domain — `raft-sintel.pth` for natural video, `raft-kitti.pth` for driving / outdoor scenes
2. Read consecutive frames with `cv2.VideoCapture` and convert to float tensors
3. Wrap inputs in `InputPadder` — RAFT requires dimensions divisible by 8
4. Run with `iters=20` for highest accuracy, `iters=12` for 1.6x faster inference at minor quality cost
5. Cache the flow alongside the original frames; recomputing per epoch is the dominant training cost
6. Either stack as 5-channel input (RGB + flow_x + flow_y) or compute magnitude `np.linalg.norm(flow, axis=-1)` for a single-channel motion map

## Key Decisions

- **Pretrained, not trained from scratch**: RAFT needs millions of synthetic flow pairs to train; transfer always wins on real video.
- **`small=False`**: the small model is 5x faster but ~30% worse on fast motion; not worth it for offline preprocessing.
- **Cache aggressively**: flow is 8 bytes/pixel and rebuilds slowly; store as `.npy` files keyed by `(video, frame)`.
- **Backward flow if needed**: call with `(f2, f1)` for backward flow — useful for occlusion handling.
- **`InputPadder` not manual padding**: RAFT's stride-8 requirement varies by resolution; the helper handles it.
- **`iters=20` at inference, not training**: training the full RAFT downstream is rare; use it as a frozen feature extractor.

## References

- [Optical flow estimation using RAFT](https://www.kaggle.com/competitions/nfl-impact-detection)
