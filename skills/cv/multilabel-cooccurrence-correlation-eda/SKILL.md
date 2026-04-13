---
name: cv-multilabel-cooccurrence-correlation-eda
description: For multi-label classification, compute the per-class binary correlation matrix restricted to multi-label rows and the conditional class counts given a rare anchor class — reveals label groupings the model can exploit (shared classifier heads, hierarchical loss weighting, post-hoc consistency rules)
---

## Overview

Multi-label problems hide a label-graph: classes that always appear together, classes that mutually exclude, and classes that anchor others. A vanilla `df.corr()` over the indicator columns is dominated by the empty-row baseline and shows nothing. Restricting to rows with `n_labels > 1`, then computing correlation, surfaces real co-occurrence. Pair this with per-rare-class conditional counts (`P(label_j | label_i)` for each rare `i`) to find which common classes systematically accompany each rare one. The output of this analysis seeds feature-engineering decisions: which classes to merge into a shared head, which to weight in a hierarchical loss, which to enforce as mutually-exclusive at post-processing.

## Quick Start

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 1) Restrict to multi-label rows and compute correlation
multi = train_labels[train_labels['number_of_targets'] > 1]
corr = multi[class_cols].corr()
sns.heatmap(corr, cmap='RdYlBu', vmin=-1, vmax=1, square=True)
plt.title('Multi-label co-occurrence correlation')
plt.show()

# 2) Conditional counts: which classes accompany a rare anchor?
def cooccur_with(anchor, df, class_cols):
    sub = df[df[anchor] == 1][class_cols].sum(axis=0)
    return sub[sub > 0].sort_values(ascending=False)

print(cooccur_with('Rods & rings', train_labels, class_cols))
print(cooccur_with('Microtubule ends', train_labels, class_cols))
```

## Workflow

1. Add an `n_labels` column counting positive labels per row
2. Filter to rows with `n_labels >= 2` for the correlation pass
3. Compute and plot the correlation heatmap; cluster rows/cols with `clustermap` to surface groups visually
4. For each rare class, list the top-K classes it co-occurs with — note anchors
5. Use the discovered groups to: (a) add weighted multi-task heads, (b) introduce a hierarchical co-occurrence loss term, (c) write post-hoc rules ("if `X`, then never predict `Y`")
6. Re-run after model training to compare prediction co-occurrence vs. ground-truth — large divergence flags a missing constraint

## Key Decisions

- **Filter to multi-label rows before correlating**: empty-row mass makes everything look anti-correlated otherwise.
- **Use `clustermap` not `heatmap` for >15 classes**: hierarchical clustering surfaces the groups automatically.
- **Conditional counts beat correlation for rare classes**: with 5 positives, correlation is unstable; raw counts are interpretable.
- **Don't infer causation**: co-occurrence reflects biology / dataset construction, not predictive direction.
- **Compare to predicted co-occurrence post-training**: divergence tells you if the model is honoring the label structure.
- **Generalizes**: any multi-label problem (audio tagging, ICD-10 medical coding, multi-genre recsys) benefits from the same EDA pass.

## References

- [Protein Atlas - Exploration and Baseline](https://www.kaggle.com/code/allunia/protein-atlas-exploration-and-baseline)
