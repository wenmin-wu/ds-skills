---
name: cv-hair-overlay-augmentation
description: Overlay real hair PNGs (masked via threshold) onto dermoscopy images to simulate body-hair occlusion as a domain-specific augmentation
---

## Overview

Dermoscopy images of skin lesions often have body hair obscuring the lesion. Training images without hair do not generalize well to test images with hair — the model learns hair is OOD. Standard cutout/cutmix injects synthetic noise, but real hair has color, shape, and directionality. The dermoscopy-specific fix is to maintain a small library of real hair PNGs (foreground on a dark background), randomly pick a few per training image, threshold them into masks, and composite onto the dermoscopy image via OpenCV `bitwise_and`. Lift of ~0.5-1 AUC points on melanoma classification reported in top Kaggle solutions.

## Quick Start

```python
import cv2, os, random
import numpy as np

class HairOverlay:
    def __init__(self, hairs_folder, max_hairs=5, p=0.5):
        self.hairs_folder = hairs_folder
        self.max_hairs = max_hairs
        self.p = p
        self.files = [f for f in os.listdir(hairs_folder) if f.endswith('.png')]

    def __call__(self, img):
        if random.random() > self.p:
            return img
        n = random.randint(0, self.max_hairs)
        for _ in range(n):
            hair = cv2.imread(os.path.join(self.hairs_folder, random.choice(self.files)))
            hair = cv2.flip(hair, random.choice([-1, 0, 1]))
            hair = cv2.rotate(hair, random.choice([0, 1, 2]))
            h, w, _ = hair.shape
            if h >= img.shape[0] or w >= img.shape[1]:
                continue
            y = random.randint(0, img.shape[0] - h)
            x = random.randint(0, img.shape[1] - w)
            roi = img[y:y+h, x:x+w]
            gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
            fg = cv2.bitwise_and(hair, hair, mask=mask)
            img[y:y+h, x:x+w] = cv2.add(bg, fg)
        return img
```

## Workflow

1. Curate 30-100 hair PNGs (e.g. extracted from other dermoscopy images via thresholding)
2. Wrap in a callable augmentation class and place in the train-only transform pipeline
3. Randomly flip/rotate each hair sprite so the model doesn't memorize fixed patterns
4. Threshold at gray > 10 to get the hair mask; composite foreground over background
5. Apply with p ≈ 0.5 — too high and the model overfits to "lots of hair = benign"

## Key Decisions

- **Real > synthetic**: procedural hair (random thin curves) doesn't match the color distribution and pretraining doesn't pick it up.
- **Random flip/rotate**: multiplies the effective hair library by ~8 and prevents orientation artifacts.
- **Threshold at low gray value**: the hair mask should include dark strands without their anti-aliased halo.
- **vs. hair removal preprocessing**: removing hair on test images is fragile; training with hair is more robust.

## References

- [Melanoma. Pytorch starter. EfficientNet](https://www.kaggle.com/code/nroman/melanoma-pytorch-starter-efficientnet)
