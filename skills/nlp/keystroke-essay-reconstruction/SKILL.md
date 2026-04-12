---
name: nlp-keystroke-essay-reconstruction
description: Replay keystroke activity logs (Input/Replace/Paste/Remove/Move) against a string buffer to reconstruct the evolving essay text
---

## Overview

When a dataset only gives you keystroke events — `activity`, `cursor_position`, `text_change` — and not the final text, you can deterministically rebuild the essay (or at least its anonymized structure) by replaying each event against a running string buffer. This unlocks a whole second feature family: sentence / paragraph / word-length statistics that are impossible to compute from event logs alone. Used in the Kaggle "Linking Writing Processes to Writing Quality" competition to add ~20-40 text-shape features on top of event-based ones.

## Quick Start

```python
def reconstruct_essay(df):
    df = df[df['activity'] != 'Nonproduction']
    out = {}
    for uid, g in df.groupby('id'):
        essay = ""
        for act, cur, txt in g[['activity','cursor_position','text_change']].values:
            if act == 'Replace':
                old, new = txt.split(' => ')
                essay = essay[:cur-len(new)] + new + essay[cur-len(new)+len(old):]
            elif act == 'Paste':
                essay = essay[:cur-len(txt)] + txt + essay[cur-len(txt):]
            elif act == 'Remove/Cut':
                essay = essay[:cur] + essay[cur+len(txt):]
            elif 'M' in act:  # Move From [a,b] To [c,d]
                pass  # splice slices per parsed offsets
            else:  # Input
                essay = essay[:cur-len(txt)] + txt + essay[cur-len(txt):]
        out[uid] = essay
    return out
```

## Workflow

1. Drop `Nonproduction` rows (mouse clicks, focus changes) — they don't modify text
2. Group events by session id and iterate in order
3. For each activity type, splice the buffer at `cursor_position` using `text_change` length
4. Keep the buffer as anonymized characters — the output is a string of `q`'s and whitespace, same shape as the real essay
5. Feed the reconstructed essay to downstream sentence/paragraph/word statistics

## Key Decisions

- **Anonymized is fine**: you only need lengths, counts, and boundaries — not content. The q-replaced text preserves everything statistical you need.
- **Replace vs. Input**: Replace carries both old and new string separated by ` => `. Parse them to compute the delta correctly, otherwise offsets drift.
- **Move is the hardest**: parse `From [a,b] To [c,d]` and splice three slices. Skip it if rare — most sessions have <1% Move events.
- **vs. text_change aggregation**: counting `text_change` values misses deletions and the final layout; only replay recovers structure.

## References

- [LGBM (X2) + NN + Fusion](https://www.kaggle.com/code/cody11null/lgbm-x2-nn-fusion)
- [Silver Bullet | Single Model | 165 Features](https://www.kaggle.com/code/mcpenguin/silver-bullet-single-model-165-features)
