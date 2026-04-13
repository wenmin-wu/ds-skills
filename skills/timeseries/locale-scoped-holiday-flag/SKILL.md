---
name: timeseries-locale-scoped-holiday-flag
description: Build a single per-row "day off" boolean from a holidays table with National/Regional/Local locale hierarchy and Work Day overrides that flip make-up working weekends back to working days
---

## Overview

Retail and transaction datasets ship with holidays tables that carry `locale` (National/Regional/Local) and `type` (Holiday/Event/Work Day/Transfer) columns. A naive `.merge` double-counts regional holidays, ignores make-up working days, and misses the fact that "Event" rows are not days off at all. The right recipe is a deterministic rule pass: seed `dayoff` from weekday, then apply National holidays globally, Regional by state match, Local by city match, and finally flip Work Day rows back to working so make-up Saturdays are modeled correctly. One clean boolean feeds your forecaster. Used in the Favorita Grocery Sales Forecasting top kernels.

## Quick Start

```python
sales['dayoff'] = sales['day'].isin([6, 7])  # Sat/Sun seed

holidays = holidays[holidays['type'] != 'Event']  # events are not days off

for d, t, locale, name in zip(holidays.date, holidays.type,
                              holidays.locale, holidays.locale_name):
    mask = (sales.date == d)
    if t != 'Work Day':
        if locale == 'National':
            sales.loc[mask, 'dayoff'] = True
        elif locale == 'Regional':
            sales.loc[mask & (sales.state == name), 'dayoff'] = True
        else:  # Local
            sales.loc[mask & (sales.city == name), 'dayoff'] = True
    else:
        sales.loc[mask, 'dayoff'] = False  # make-up working day
```

## Workflow

1. Seed `dayoff` from `day_of_week in {Sat, Sun}`
2. Drop rows where `type == 'Event'` — events are not days off
3. Walk each remaining holiday row and apply National globally, Regional by state, Local by city
4. Apply `Work Day` rows **last** so they override any previously-set dayoff (the precedence matters)
5. Use the resulting boolean as a direct feature or as a segmentation key for per-day models

## Key Decisions

- **Drop Events up front, don't filter mid-loop**: keeps the rule-application order clean.
- **Work Day rows last**: they flip make-up working Saturdays back to working — precedence must be after all holiday types.
- **Match by `locale_name` against store `state`/`city`**: this is why regional and local holidays don't leak into other regions.
- **Single boolean vs one-hot locale**: a single `dayoff` column is leak-proof when a date has overlapping entries; one-hot locale flags double-count.

## References

- [A first Kaggle - Part 1 - Forecasting store #47](https://www.kaggle.com/code/jagangupta/a-first-kaggle-part-1-forecasting-store-47)
