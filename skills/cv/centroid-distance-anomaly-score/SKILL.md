---
name: cv-centroid-distance-anomaly-score
description: Score each face in a video as anomalous by computing L2 distance from the embedding centroid of all faces, then convert to probability via logistic function
---

# Centroid Distance Anomaly Score

## Overview

In a real video, all face embeddings should cluster tightly (same person, consistent appearance). Deepfakes introduce subtle inconsistencies that push some embeddings away from the centroid. Computing the L2 distance from each frame's face embedding to the mean embedding, then averaging and passing through a logistic function, produces a calibrated deepfake probability without needing labeled training data.

## Quick Start

```python
import numpy as np

def centroid_anomaly_score(embeddings, bias=-0.3, weight=0.7):
    centroid = embeddings.mean(axis=0)
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    mean_dist = distances.mean()
    prob = 1.0 / (1.0 + np.exp(-(bias + weight * mean_dist)))
    return prob

# embeddings: (N_frames, 512) from a pretrained face model
prob_fake = centroid_anomaly_score(face_embeddings)
```

## Workflow

1. Extract face crops from N sampled video frames
2. Compute face embeddings using a pretrained model (InceptionResnetV1, ArcFace)
3. Calculate the centroid (mean) of all embeddings
4. Compute L2 distance from each embedding to the centroid
5. Average the distances and pass through a logistic function
6. Output is a probability: higher = more likely deepfake

## Key Decisions

- **Embedding model**: VGGFace2-pretrained InceptionResnetV1 gives strong baselines; ArcFace is more discriminative
- **Logistic calibration**: fit bias and weight on a validation set; defaults (-0.3, 0.7) are a starting point
- **Outlier frames**: some frames may have detection failures — filter by face confidence before embedding
- **Multi-face videos**: compute per-face-track centroids separately, average scores
- **vs. supervised**: centroid distance is unsupervised and doesn't need fake examples for training — useful as a feature in an ensemble

## References

- [Facial recognition model in pytorch](https://www.kaggle.com/code/timesler/facial-recognition-model-in-pytorch)
