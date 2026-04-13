---
name: cv-dot-annotation-blob-diff-extraction
description: Recover (x, y, class) point labels from color-coded dot-annotation image pairs via absdiff + blackout masking + Laplacian-of-Gaussian blob detection + center-pixel RGB classification
---

## Overview

Many wildlife/cell/object-counting datasets ship annotations as a *second copy of the image* with colored dots painted over each instance. You have to recover the point list yourself. The canonical recipe is four steps: `cv2.absdiff(dotted, raw)` to isolate the dots, bitwise-mask out any blacked-out exclusion regions present in either image, run `skimage.feature.blob_log` with tight sigma bounds matched to the known dot radius, then classify each blob by reading the centroid pixel color from the *dotted* image (not the diff — the diff desaturates the color). Used in NOAA Steller Sea Lion Population Count top kernels.

## Quick Start

```python
import cv2, numpy as np, skimage.feature

img_raw = cv2.imread(raw_path)
img_dot = cv2.imread(dotted_path)
diff = cv2.absdiff(img_dot, img_raw)

# mask out blacked-out regions (annotator exclusions) from either image
m1 = cv2.cvtColor(img_dot, cv2.COLOR_BGR2GRAY); m1[m1 < 20] = 0; m1[m1 > 0] = 255
m2 = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY); m2[m2 < 20] = 0; m2[m2 > 0] = 255
diff = cv2.bitwise_or(diff, diff, mask=m1)
diff = cv2.bitwise_or(diff, diff, mask=m2)

gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
blobs = skimage.feature.blob_log(gray, min_sigma=3, max_sigma=4, num_sigma=1, threshold=0.02)

points = []
for y, x, _ in blobs:
    b, g, r = img_dot[int(y), int(x)]
    if   r > 200 and b < 50  and g < 50:              cls = 'adult_male'
    elif r > 200 and b > 200 and g < 50:              cls = 'subadult_male'
    elif r < 100 and g > 100 and b < 100:             cls = 'juvenile'
    elif r < 100 and g < 100 and 150 < b < 200:       cls = 'pup'
    elif r < 150 and g < 50  and b < 100:             cls = 'adult_female'
    else:                                              continue
    points.append((int(x), int(y), cls))
```

## Workflow

1. Compute `cv2.absdiff(dotted, raw)` to leave only the dot pixels
2. Build blackout masks from both images by thresholding grayscale < 20 and bitwise-OR the diff through them
3. Convert to grayscale and run `blob_log` with `min/max_sigma` bracketing the expected dot radius and `num_sigma=1` for speed
4. For each blob centroid, sample the BGR pixel from the **original dotted image** (preserves saturation)
5. Route the pixel through a hand-tuned RGB decision tree; drop fall-throughs to an `error` bucket to quantify noise

## Key Decisions

- **absdiff, not signed subtract**: color-agnostic and symmetric across image order.
- **Mask from BOTH images**: annotators black out regions in either copy; missing either mask leaks false blobs.
- **`num_sigma=1` with tight sigma range**: single-scale LoG is fast and precise when dot radius is fixed.
- **Sample color from `img_dot`, not `diff`**: the diff fades saturated colors, wrecking the decision tree.
- **Explicit `error` fallback**: catches out-of-gamut dots and quantifies labeling noise rather than silently mislabeling.

## References

- [Use keras to classify Sea Lions: 0.91 accuracy](https://www.kaggle.com/code/outrunner/use-keras-to-classify-sea-lions-0-91-accuracy)
- [Get coordinates using blob detection](https://www.kaggle.com/code/radustoicescu/get-coordinates-using-blob-detection)
