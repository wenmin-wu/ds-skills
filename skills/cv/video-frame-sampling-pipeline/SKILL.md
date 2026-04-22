---
name: cv-video-frame-sampling-pipeline
description: Efficiently sample N evenly-spaced frames from a video using OpenCV grab/retrieve pattern with optional resize for batch face detection or classification
---

# Video Frame Sampling Pipeline

## Overview

Processing every frame of a video is wasteful for classification tasks. Sampling N evenly-spaced frames via `np.linspace` and using OpenCV's `grab()`/`retrieve()` pattern (grab skips decoding, retrieve decodes only selected frames) is 3-5x faster than reading every frame. This produces a fixed-size batch suitable for CNN inference or face detection.

## Quick Start

```python
import cv2
import numpy as np
from PIL import Image

def sample_frames(video_path, n_frames=17, resize=None):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, n_frames).astype(int)
    indices_set = set(indices)
    frames = []
    for i in range(total):
        grabbed = cap.grab()
        if not grabbed:
            break
        if i in indices_set:
            ret, frame = cap.retrieve()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                if resize:
                    img = img.resize([int(d * resize) for d in img.size])
                frames.append(img)
    cap.release()
    return frames
```

## Workflow

1. Open video with `cv2.VideoCapture`, read total frame count
2. Compute N evenly-spaced indices via `np.linspace(0, total-1, N)`
3. Loop through all frames using `grab()` (fast, no decode)
4. Call `retrieve()` only for selected indices (decodes the frame)
5. Convert BGR→RGB, optionally resize, collect into a list
6. Pass the frame batch to a face detector or classifier

## Key Decisions

- **N frames**: 15-20 for deepfake detection; 1-5 for thumbnail/preview tasks
- **grab vs read**: `grab()` without `retrieve()` skips decoding — essential for long videos
- **Resize factor**: 0.25-0.5 for face detection preprocessing; full resolution for final classification
- **Edge handling**: if total < N, use all available frames and pad with duplicates
- **vs. seek**: `cap.set(cv2.CAP_PROP_POS_FRAMES, i)` is unreliable for some codecs; grab/retrieve is more robust

## References

- [Facial recognition model in pytorch](https://www.kaggle.com/code/timesler/facial-recognition-model-in-pytorch)
