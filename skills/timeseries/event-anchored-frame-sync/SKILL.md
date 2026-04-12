---
name: timeseries-event-anchored-frame-sync
description: Align low-Hz sensor data to high-fps video by anchoring a named event (e.g. ball_snap) to a known frame index and converting time offsets via fps
---

## Overview

Sensor and video streams are rarely recorded with perfectly synchronized clocks. In the NFL Helmet Assignment competition, player tracking (NGS) is 10Hz with wall-clock timestamps while video is ~60fps counted from frame 0 — you cannot join them by timestamp. The trick is that both streams share a discrete labeled event: the `ball_snap`. If you know which video frame contains the snap (a constant per dataset), you can anchor the NGS row for `event == 'ball_snap'` to that frame, compute a `snap_offset` in seconds, multiply by fps, and derive `est_frame` for every NGS row. Result: a deterministic join key between the two streams with millisecond-level accuracy.

## Quick Start

```python
def add_est_frame(tracks, fps=59.94, snap_frame=10):
    tracks = tracks.copy()
    tracks['game_play'] = (tracks['gameKey'].astype(str) + '_'
                           + tracks['playID'].astype(str).str.zfill(6))
    tracks['time'] = pd.to_datetime(tracks['time'])

    # Anchor: for each play, find the wall-clock time of the ball_snap row
    snap_time = (tracks.query('event == "ball_snap"')
                 .groupby('game_play')['time'].first().to_dict())
    tracks['snap'] = tracks['game_play'].map(snap_time)

    # Offset from snap in seconds, then convert to frame index
    tracks['snap_offset'] = (
        (tracks['time'] - tracks['snap']).dt.total_seconds()
    )
    tracks['est_frame'] = (
        (tracks['snap_offset'] * fps) + snap_frame
    ).round().astype(int)
    return tracks
```

## Workflow

1. Identify a discrete event that appears in both streams (ball_snap, first-touch, shutter release)
2. Pick one row per session where `event == <anchor>` in the sensor stream
3. Find the known video frame index of the anchor event (often a constant from dataset metadata)
4. Compute `offset_sec = sensor_time - anchor_time` for every sensor row
5. Convert to `est_frame = round(offset_sec * fps + anchor_frame)`

## Key Decisions

- **Anchor event must be discrete and unique**: events that repeat (e.g. "whistle") break the alignment.
- **Watch for fps fractions**: NTSC video is 59.94 not 60; using integer fps drifts ~0.1% over long clips.
- **Nearest-neighbor lookup downstream**: even after sync, multiple sensor rows may fall between frames — use `find_nearest(est_frames, video_frame)` to pick one.
- **vs. timestamp-only join**: wall-clock timestamps drift between recorders; event anchoring cancels the absolute offset entirely.

## References

- [NFL Helmet Assignment - Getting Started Guide](https://www.kaggle.com/code/robikscube/nfl-helmet-assignment-getting-started-guide)
- [NFL Baseline - Simple Helmet Mapping](https://www.kaggle.com/code/its7171/nfl-baseline-simple-helmet-mapping)
