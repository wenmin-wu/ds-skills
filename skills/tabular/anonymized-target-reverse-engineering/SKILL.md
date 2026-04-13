---
name: tabular-anonymized-target-reverse-engineering
description: Probe an anonymized regression target by testing whether simple invertible transforms (2**y, exp(y), log(y), affine rescale) produce a distribution with recognizable structure — round numbers, integer histograms, or a familiar finance/retail range — and use the recovered semantics to motivate features and loss choices the host's bland description would never suggest
---

## Overview

When a competition host hides the target ("loyalty score, normalized") your first move is not to build a model — it's to figure out what the number *actually* means. Histogram it; try `2**y`, `np.exp(y)`, `y * std + mean` for various candidate scales. If a transform makes the distribution snap onto integer values, dollar amounts, or a known KPI range, you have just unlocked feature engineering the host did not intend to share. On Elo, applying `2**target` revealed an integer-like structure consistent with a churn-month count; the discovery motivated month-recency features that pushed solutions up the LB. This is dual-use: also use it to detect leakage (if the transform recovers a column you already have).

## Quick Start

```python
import numpy as np, matplotlib.pyplot as plt

y = train.target.values

# 1) raw histogram
plt.hist(y, bins=200); plt.title('raw'); plt.show()

# 2) try common inverse transforms
for name, t in [('2**y', 2**y), ('exp(y)', np.exp(y)),
                ('y**2', y**2), ('1/(1-y)', 1/(1-y))]:
    vals = t[np.isfinite(t)]
    plt.hist(vals, bins=200); plt.title(name); plt.show()
    print(name, 'min=', vals.min(), 'max=', vals.max(),
          'frac_int=', np.mean(np.isclose(vals, np.round(vals), atol=1e-3)))
```

## Workflow

1. Plot the raw target — note any spikes, gaps, hard floors/ceilings (sentinel values are clues)
2. Try the standard transform menu: `exp`, `log`, `2**`, `**2`, affine rescales using `train.describe()` percentiles as anchors
3. For each transform compute: range, fraction-near-integer, mode, top-10 most-frequent values
4. If a transform yields integer-like or round-number structure, search the host's data for a column whose values match
5. Cross-check on a small sample by hand — if `2**y == months_since_last_purchase` for 5 random cards, you've recovered the target
6. Use the discovered semantics to design features (e.g., monthly aggregates if the target is monthly)

## Key Decisions

- **Spikes are tells, not noise**: a single huge spike usually points to a sentinel value or a clipped boundary; both reveal target semantics.
- **Try inverses of the obvious normalizations**: hosts most often use z-score, min-max, or log — invert each.
- **Stop when you have a hypothesis, not a proof**: even a *plausible* meaning is enough to seed feature engineering; certainty isn't required.
- **Don't post the discovery**: if it's exploitable you want to keep it, if it's leakage report it to the host.
- **Combine with adversarial validation**: if your transform recovers a feature train and test disagree on, that's a leak hint.
- **Beware of overfitting to the discovery**: validate that features motivated by the recovered semantics actually generalize.

## References

- [Target — true meaning revealed (Elo Merchant)](https://www.kaggle.com/competitions/elo-merchant-category-recommendation)
