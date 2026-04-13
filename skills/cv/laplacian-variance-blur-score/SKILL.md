---
name: cv-laplacian-variance-blur-score
description: Score image sharpness with the variance of the Laplacian (Pech-Pacheco) as a single scalar feature for downstream tabular models or as a hard blur filter
---

## Overview

Blurry photos are a universal quality problem in classified-ads, product-catalog, and user-upload datasets. The Pech-Pacheco method — variance of the Laplacian on the grayscale image — is the canonical one-liner: low variance means a flat second-derivative response (the image has no crisp edges, i.e. blurry); high variance means sharp transitions. You can either feed the raw score as a regression feature or threshold it (~100 is the textbook starting point) to binary-flag blurry listings. Used alongside dullness, whiteness, and edge-density scores to build a compact image-quality feature block on Avito Demand Prediction.

## Quick Start

```python
import cv2

def blur_score(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def is_blurry(path, threshold=100.0):
    return blur_score(path) < threshold
```

## Workflow

1. Load the image with cv2 and convert to grayscale (`COLOR_BGR2GRAY`)
2. Apply `cv2.Laplacian(gray, cv2.CV_64F)` — **CV_64F is critical**, uint8 clips negative second-derivative values and destroys the signal
3. Call `.var()` on the Laplacian response
4. Use the raw score as a numeric feature OR threshold to produce a binary `is_blurry` flag
5. Calibrate the threshold per-dataset by sorting scores ascending and eyeballing the low-score tail

## Key Decisions

- **CV_64F vs CV_8U**: uint8 clips negatives and the variance collapses. Always use float.
- **Grayscale first**: Laplacian on RGB runs three times and adds no signal for blur.
- **Variance, not mean absolute**: per the original Pech-Pacheco paper, variance is the discriminator.
- **Threshold is dataset-dependent**: 100 is a starting point for phone-camera data; surveillance or dermoscopy need their own calibration.

## References

- [Ideas for Image Features and Image Quality](https://www.kaggle.com/code/shivamb/ideas-for-image-features-and-image-quality)
- Pech-Pacheco et al., *Diatom autofocusing in brightfield microscopy: a comparative study*, ICPR 2000
