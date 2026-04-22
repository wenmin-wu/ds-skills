---
name: cv-frame-prediction-averaging
description: Average per-frame sigmoid predictions across sampled video frames to produce a stable video-level classification probability
---

# Frame Prediction Averaging

## Overview

Video classification models often process individual frames independently. Averaging the sigmoid outputs across all sampled frames reduces noise from individual frame variability (occlusion, motion blur, detection failures) and produces a more stable video-level prediction. This is simpler and often competitive with temporal models like LSTMs for binary classification tasks.

## Quick Start

```python
import torch
import numpy as np

def predict_video(model, frames, device="cuda"):
    model.eval()
    batch = torch.stack(frames).to(device)
    with torch.no_grad():
        logits = model(batch).squeeze()
        probs = torch.sigmoid(logits)
    return probs.mean().item()

# frames: list of N preprocessed tensors from sampled video frames
video_prob = predict_video(model, face_tensors)
```

## Workflow

1. Sample N frames from the video (typically 15-20)
2. Extract and preprocess face crops from each frame
3. Run each frame through the classifier to get sigmoid probabilities
4. Average all per-frame probabilities for the final video-level score
5. Optionally clip to [0.05, 0.95] to avoid extreme predictions from single-frame errors

## Key Decisions

- **Averaging vs. max**: mean is robust to outliers; max captures the single most suspicious frame (higher recall, lower precision)
- **Weighted average**: weight frames by face detection confidence to downweight low-quality detections
- **Frames with no face**: exclude from average rather than using a default value
- **Clipping**: clip final probability to [0.01, 0.99] to avoid log-loss penalties from extreme values
- **vs. temporal models**: averaging ignores frame order but is more robust with limited training data

## References

- [Inference Demo](https://www.kaggle.com/code/humananalog/inference-demo)
