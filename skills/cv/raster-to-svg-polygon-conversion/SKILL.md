---
name: cv-raster-to-svg-polygon-conversion
description: >
  Converts a raster image to a size-bounded SVG via K-means color quantization, contour extraction, importance-ranked polygon assembly, and progressive simplification.
---
# Raster-to-SVG Polygon Conversion

## Overview

Converting bitmaps to SVGs enables resolution-independent rendering, but naive tracing produces files too large for size-constrained submissions or web use. This technique quantizes colors with K-means, extracts contours per color, ranks polygons by a composite importance score (area, centrality, complexity), then greedily assembles SVG polygons until a byte budget is hit. A second pass fills remaining budget with progressively simplified versions of skipped polygons. This produces compact, visually faithful SVGs within strict size limits.

## Quick Start

```python
import cv2
import numpy as np

def bitmap_to_svg(image, max_bytes=10000, num_colors=12):
    img = np.array(image)
    pixels = img.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10,
                                     cv2.KMEANS_RANDOM_CENTERS)
    quantized = centers[labels.flatten()].reshape(img.shape).astype(np.uint8)

    h, w = img.shape[:2]
    features = []
    for color in np.unique(centers.astype(np.uint8), axis=0):
        mask = cv2.inRange(quantized, color, color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50: continue
            eps = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True)
            M = cv2.moments(cnt)
            cx = int(M['m10'] / (M['m00'] + 1e-5))
            cy = int(M['m01'] / (M['m00'] + 1e-5))
            dist = np.sqrt(((cx - w/2) / w)**2 + ((cy - h/2) / h)**2)
            importance = area * (1 - dist) / (len(approx) + 1)
            pts = ' '.join(f'{p[0][0]},{p[0][1]}' for p in approx)
            hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
            features.append({'points': pts, 'color': hex_color, 'importance': importance})

    features.sort(key=lambda f: f['importance'], reverse=True)
    svg = f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">\n'
    for f in features:
        line = f'<polygon points="{f["points"]}" fill="{f["color"]}"/>\n'
        if len((svg + line + '</svg>').encode()) > max_bytes: break
        svg += line
    return svg + '</svg>'
```

## Workflow

1. Quantize image colors with K-means (8-16 clusters)
2. Extract contours per quantized color
3. Simplify contours with `approxPolyDP`
4. Rank by importance: area * centrality / complexity
5. Greedily add polygons until byte budget is reached

## Key Decisions

- **num_colors**: 8-12 for compact SVGs; 16-24 for higher fidelity
- **Simplification epsilon**: 0.02 of arc length is a good default; lower for more detail
- **Importance formula**: Balances visual impact (area), focus (centrality), and efficiency (complexity)
- **Progressive simplification**: Second pass with higher epsilon fills remaining budget

## References

- [Stable Diffusion -> SVG -> Scoring Metric](https://www.kaggle.com/code/richolson/stable-diffusion-svg-scoring-metric)
