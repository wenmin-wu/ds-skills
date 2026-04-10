---
name: cv-morphological-lung-segmentation
description: Segment lung regions from CT using HU thresholding, connected-component labeling, and morphological opening
domain: cv
---

# Morphological Lung Segmentation

## Overview

Isolate lung tissue from CT volumes without deep learning. Threshold HU values to separate air-filled lung from dense tissue, use connected-component labeling to identify background regions touching image borders, then apply morphological opening to clean boundaries. Useful as a preprocessing mask before feeding lung ROIs into a model.

## Quick Start

```python
import numpy as np
from skimage import measure, morphology
from skimage.morphology import disk

def segment_lung_mask(image, threshold=-320):
    """Segment lungs from a 3D HU volume.
    
    Args:
        image: (D, H, W) array in Hounsfield Units
        threshold: HU cutoff (air vs tissue boundary)
    Returns:
        masked volume with non-lung regions zeroed out
    """
    segmented = np.zeros(image.shape)
    for n in range(image.shape[0]):
        binary = np.array(image[n] > threshold, dtype=np.int8) + 1
        labels = measure.label(binary)
        # Remove regions touching borders (background)
        border_labels = np.unique(np.concatenate([
            labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]
        ]))
        for bl in border_labels:
            binary[labels == bl] = 2
        binary = morphology.opening(binary, disk(2))
        binary = 1 - (binary - 1)  # invert: lung=1, background=0
        segmented[n] = binary * image[n]
    return segmented

mask = segment_lung_mask(hu_volume)
```

## Key Decisions

- **Threshold at −320 HU**: separates aerated lung from soft tissue; adjust for emphysema
- **Border-touching removal**: background air outside the body shares HU with lung — border check distinguishes them
- **Morphological opening**: disk(2) removes small noise without eroding lung boundaries
- **Per-slice processing**: avoids 3D connected-component overhead; works on anisotropic volumes

## References

- Source: [pulmonary-dicom-preprocessing](https://www.kaggle.com/code/allunia/pulmonary-dicom-preprocessing)
- Competition: SIIM-FISABIO-RSNA COVID-19 Detection
