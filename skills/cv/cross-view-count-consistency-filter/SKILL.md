---
name: cv-cross-view-count-consistency-filter
description: Drop false-positive detections by requiring matched per-frame box counts across paired synchronized camera views (Endzone vs Sideline) — if both views see the same scene at the same instant, true events should appear in both
---

## Overview

Multi-camera event detection (NFL impacts, surveillance, sports tracking) provides a free post-processing filter: when two cameras observe the same play from different angles at the same moment, every real event should produce detections in *both* views. False positives are usually idiosyncratic to a single camera (compression artifact, motion blur in one view only), so frames where the two views disagree on the box count are almost certainly contaminated. This is a one-pass groupby-and-drop that costs nothing but throws away a meaningful chunk of FPs without touching the detector.

## Quick Start

```python
import pandas as pd

drop_idx = []
for (gk, pid), _ in test_df.groupby(['gameKey', 'playID']):
    play = test_df.query('gameKey == @gk and playID == @pid')
    for idx, row in play.iterrows():
        f = row['frame']
        n_side = play.query('view == "Sideline" and frame == @f').shape[0]
        n_end  = play.query('view == "Endzone"  and frame == @f').shape[0]
        if n_side != n_end:
            drop_idx.append(idx)

test_df = test_df.drop(index=drop_idx).reset_index(drop=True)
```

## Workflow

1. Group detections by `(gameKey, playID)` — the unit of synchronized capture
2. For each frame within the play, count detections in each view
3. If the per-view counts differ, drop *all* detections in that frame (not just the extras)
4. Apply this *after* the per-view detector inference but *before* score-based filtering
5. Optionally allow a tolerance of ±1 if your detector is noisy

## Key Decisions

- **Drop the whole frame, not the surplus**: you can't tell which of N detections is the spurious one — if the count is wrong the frame is suspect end-to-end.
- **Per-frame, not per-play**: a real impact at frame 100 shouldn't be dropped because frame 200 is asymmetric.
- **Strict equality, no tolerance**: synchronized cameras at 60fps are well-aligned; a tolerance often hides real disagreements.
- **vs. cross-view IoU matching**: counting is O(1) per frame; geometric matching needs camera calibration that's usually not provided.
- **Only useful when both views are detector inputs**: if you only have one camera, this trick doesn't apply.

## References

- [Both zones 2Class Object Detection strict filter](https://www.kaggle.com/competitions/nfl-impact-detection)
