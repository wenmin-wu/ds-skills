---
name: llm-svg-constrained-generation
description: >
  Prompts an LLM to generate valid SVG by embedding an explicit element/attribute allowlist and a one-shot example, then extracts the last valid SVG block from output.
---
# SVG Constrained Generation

## Overview

LLMs can generate SVG code, but without constraints they produce invalid elements, inline styles, scripts, or malformed paths. This technique embeds the exact allowed elements and attributes directly in the prompt as a constraint block, provides one concrete example, and ends the prompt mid-tag so the model continues inside a valid SVG root. After generation, regex extracts the last `<svg>...</svg>` block. This pattern generalizes to any structured markup where you need to enforce a schema via prompting.

## Quick Start

```python
import re

PROMPT = """Generate SVG code for the following description.

<constraints>
* Allowed elements: svg, path, circle, rect, ellipse, line, polygon, polyline, g
* Allowed attributes: viewBox, width, height, d, cx, cy, r, x, y, fill, stroke, stroke-width, points, transform
* No text, no style tags, no scripts
</constraints>

<example>
Description: "A red circle"
```svg
<svg viewBox="0 0 256 256" width="256" height="256">
  <circle cx="128" cy="128" r="80" fill="red"/>
</svg>
```
</example>

Description: "{description}"
```svg
<svg viewBox="0 0 256 256" width="256" height="256">
"""

def extract_svg(output):
    matches = re.findall(r'<svg.*?</svg>', output, re.DOTALL | re.IGNORECASE)
    return matches[-1] if matches else DEFAULT_SVG

svg = extract_svg(model.generate(PROMPT.format(description=desc)))
```

## Workflow

1. Define allowed elements and attributes as an explicit constraint block
2. Include one concrete example showing the expected format
3. End prompt mid-tag to force continuation inside valid SVG
4. Generate with the LLM
5. Extract last `<svg>...</svg>` match via regex

## Key Decisions

- **Allowlist over blocklist**: Explicitly listing allowed elements is safer than trying to block bad ones
- **Last match**: Take the last SVG block — models often produce preamble or multiple attempts
- **Fallback**: Always have a default SVG for when extraction fails
- **Generalizes to**: HTML fragments, LaTeX, JSON schemas — any structured output with constraints

## References

- [Drawing with LLMs - Getting Started with Gemma 2](https://www.kaggle.com/code/ryanholbrook/drawing-with-llms-getting-started-with-gemma-2)
