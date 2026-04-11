---
name: cv-local-contrast-enhancement
description: >
  Subtracts a Gaussian-blurred version of the image from itself to normalize local illumination and enhance fine structural details.
---
# Local Contrast Enhancement

## Overview

Images with uneven illumination (retinal fundus, microscopy, satellite) have regions where detail is lost in bright spots or dark shadows. Subtracting a heavily blurred version of the image from the original removes the low-frequency illumination gradient while preserving high-frequency details like edges and textures. The weighted additive blend `4*image - 4*blur + 128` centers the output at mid-gray with enhanced local contrast.

## Quick Start

```python
import cv2

def enhance_local_contrast(image, sigma=10):
    """Enhance local contrast by subtracting Gaussian blur.

    Args:
        image: (H, W, 3) uint8 RGB image
        sigma: Gaussian kernel sigma; larger = removes broader gradients
    Returns:
        (H, W, 3) uint8 contrast-enhanced image
    """
    return cv2.addWeighted(
        image, 4,
        cv2.GaussianBlur(image, (0, 0), sigma),
        -4, 128
    )

# Usage in preprocessing pipeline
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (512, 512))
img = enhance_local_contrast(img, sigma=10)
```

## Workflow

1. Load and resize image to target dimensions
2. Compute Gaussian blur with chosen sigma
3. Blend: `4 * original - 4 * blurred + 128`
4. Result has normalized illumination with enhanced local detail

## Key Decisions

- **sigma**: 10 for retinal images; increase for larger images or broader illumination gradients
- **Weight 4**: Higher weight = stronger contrast; 2-4 is typical range
- **Offset 128**: Centers output at mid-gray; adjust if using different normalization
- **Before/after resize**: Apply after resize for consistent sigma effect across scales
- **Use cases**: Retinal imaging, histology, satellite imagery, any uneven illumination

## References

- [APTOS Eye Preprocessing in Diabetic Retinopathy](https://www.kaggle.com/code/ratthachat/aptos-eye-preprocessing-in-diabetic-retinopathy)
