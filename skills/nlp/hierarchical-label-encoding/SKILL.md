---
name: nlp-hierarchical-label-encoding
description: Concatenate multi-level categorical fields into a compound label (Category:Subcategory) for flat multiclass classification
domain: nlp
---

# Hierarchical Label Encoding

## Overview

When labels have a hierarchical structure (Category → Subcategory → Specific), concatenate levels into a single compound string and LabelEncode for flat multiclass classification. Simpler than hierarchical classifiers, and models can learn the structure implicitly. Decode back to original levels at inference via string splitting.

## Quick Start

```python
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Encode: create compound labels
df['compound_label'] = df['Category'] + ':' + df['Subcategory']
le = LabelEncoder()
df['label'] = le.fit_transform(df['compound_label'])
n_classes = len(le.classes_)

# Train model with n_classes output neurons...

# Decode: convert top-K predictions back to original labels
def decode_top_k(probs, le, k=3):
    """Decode top-K softmax predictions to original compound labels."""
    top_k = np.argsort(-probs, axis=1)[:, :k]
    decoded = le.inverse_transform(top_k.flatten())
    return decoded.reshape(top_k.shape)

top3_labels = decode_top_k(probs, le, k=3)
# Each row: ["Category_A:Sub_1", "Category_B:Sub_3", "Category_A:Sub_2"]
```

## Key Decisions

- **Delimiter choice**: use ":" or "|" — avoid characters that appear in label text
- **Handle missing**: fill NaN subcategories with "NA" before concatenation
- **Large label space**: works up to ~10K classes; beyond that, consider retrieval-based approaches
- **Preserves hierarchy**: model can learn that "Algebra:Sign_Error" and "Algebra:Order_Error" share a parent

## References

- Source: [modernbert-large-cv-0-938](https://www.kaggle.com/code/cdeotte/modernbert-large-cv-0-938)
- Competition: MAP - Charting Student Math Misunderstandings
