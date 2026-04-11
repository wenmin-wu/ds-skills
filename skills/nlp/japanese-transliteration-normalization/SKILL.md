---
name: nlp-japanese-transliteration-normalization
description: >
  Converts Japanese scripts (Hiragana, Katakana, Kanji) to romanized ASCII using pykakasi for cross-script entity matching.
---
# Japanese Transliteration Normalization

## Overview

Multilingual entity matching fails when the same entity appears in different scripts — "東京タワー" and "Tokyo Tower" won't match by string similarity. Pykakasi converts Japanese Hiragana, Katakana, and Kanji to romanized ASCII (romaji), enabling standard string similarity metrics to work across scripts. Apply selectively to Japanese records before computing matching features.

## Quick Start

```python
import pykakasi

def romanize_japanese(df, text_cols, country_col="country"):
    """Convert Japanese text fields to romaji for cross-script matching."""
    kks = pykakasi.kakasi()
    
    def convert_text(text):
        if not isinstance(text, str):
            return text
        result = kks.convert(text)
        return " ".join([item["hepburn"] for item in result])
    
    jp_mask = df[country_col] == "JP"
    for col in text_cols:
        df.loc[jp_mask, col] = df.loc[jp_mask, col].apply(convert_text)
    return df

# Romanize name and address fields for Japanese records
df = romanize_japanese(df, ["name", "address", "city", "state"])
```

## Workflow

1. Filter records by country/locale to identify Japanese text
2. Apply pykakasi converter to each text field
3. Use Hepburn romanization (most common standard)
4. Proceed with standard string similarity computation on romanized text

## Key Decisions

- **Romanization standard**: Hepburn (default) vs Kunrei-shiki — Hepburn is more internationally recognized
- **Selective application**: Only convert Japanese records; applying to non-Japanese text wastes time
- **Other languages**: For Chinese use pypinyin, for Korean use korean-romanizer
- **Install**: `pip install pykakasi`

## References

- [Public: 0.861 | PyKakasi & Radian Coordinates](https://www.kaggle.com/code/nlztrk/public-0-861-pykakasi-radian-coordinates)
