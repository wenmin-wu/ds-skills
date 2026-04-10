---
name: tabular-vote-ensemble-outer-join
description: Ensemble ranked recommendation lists by outer-joining exploded candidates and re-ranking by weighted vote sum
domain: tabular
---

# Vote Ensemble via Outer Join

## Overview

When ensembling multiple recommendation submissions that each predict a ranked list of item IDs, explode each list into per-candidate rows, outer-join on (session, item), sum weighted votes, and re-rank. Unlike rank averaging, this handles disjoint candidate sets naturally -- items appearing in more submissions get more votes.

## Quick Start

```python
import polars as pl

def vote_ensemble(submission_paths, weights=None, k=20):
    """Ensemble ranked lists by weighted vote counting.
    
    Args:
        submission_paths: list of CSV paths (session_type, labels)
        weights: per-submission weight (default: equal)
        k: number of items to keep per session
    """
    if weights is None:
        weights = [1] * len(submission_paths)
    
    subs = []
    for path, w in zip(submission_paths, weights):
        sub = (pl.read_csv(path)
            .with_columns(pl.col('labels').str.split(' '))
            .explode('labels')
            .with_columns([
                pl.col('labels').cast(pl.UInt32).alias('aid'),
                pl.lit(w).cast(pl.Float32).alias('vote')
            ])
            .select(['session_type', 'aid', 'vote']))
        subs.append(sub)
    
    # Outer join and sum votes
    merged = pl.concat(subs)
    ranked = (merged
        .group_by(['session_type', 'aid'])
        .agg(pl.col('vote').sum().alias('vote_sum'))
        .sort(['session_type', 'vote_sum'], descending=[False, True])
        .group_by('session_type')
        .agg(pl.col('aid').head(k).cast(pl.Utf8).alias('labels'))
        .with_columns(pl.col('labels').list.join(' ')))
    return ranked
```

## Key Decisions

- **Outer join vs inner**: outer ensures candidates unique to one submission aren't lost
- **Weighted votes**: give higher weight to better-scoring individual submissions
- **Polars for speed**: handles millions of rows efficiently; concat + group_by avoids nested joins
- **Complements rank averaging**: use vote ensemble when candidate sets differ across submissions

## References

- Source: [0-578-ensemble-of-public-notebooks](https://www.kaggle.com/code/karakasatarik/0-578-ensemble-of-public-notebooks)
- Competition: OTTO - Multi-Objective Recommender System
