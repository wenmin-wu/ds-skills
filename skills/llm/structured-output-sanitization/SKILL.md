---
name: llm-structured-output-sanitization
description: >
  Parses LLM-generated markup (SVG, HTML, XML) with lxml, strips disallowed elements and attributes via an allowlist, and validates structural constraints like path data.
---
# Structured Output Sanitization

## Overview

Even with constrained prompts, LLMs produce invalid or disallowed markup — extra attributes, unsupported elements, malformed path data. Post-generation sanitization parses the output as XML, walks the tree, removes anything not in an allowlist, validates structural properties (e.g., SVG path `d` attribute syntax), and serializes the cleaned result. This is the defense-in-depth complement to prompt-level constraints: the prompt reduces violations, sanitization eliminates them.

## Quick Start

```python
from lxml import etree
import re

ALLOWED = {
    'svg': {'viewBox', 'width', 'height', 'xmlns'},
    'circle': {'cx', 'cy', 'r', 'fill', 'stroke'},
    'rect': {'x', 'y', 'width', 'height', 'fill', 'stroke'},
    'path': {'d', 'fill', 'stroke', 'stroke-width'},
    'common': {'transform', 'opacity'},
}
PATH_RE = re.compile(r'^[MmLlHhVvCcSsQqTtAaZz0-9\s,.\-eE]+$')

def sanitize_svg(svg_string):
    parser = etree.XMLParser(remove_blank_text=True, remove_comments=True)
    root = etree.fromstring(svg_string.encode(), parser=parser)
    to_remove = []
    for el in root.iter():
        tag = etree.QName(el.tag).localname
        if tag not in ALLOWED:
            to_remove.append(el); continue
        allowed_attrs = ALLOWED[tag] | ALLOWED.get('common', set())
        for attr in list(el.attrib):
            if etree.QName(attr).localname not in allowed_attrs:
                del el.attrib[attr]
        if tag == 'path' and not PATH_RE.match(el.get('d', '')):
            to_remove.append(el)
    for el in to_remove:
        if el.getparent() is not None:
            el.getparent().remove(el)
    return etree.tostring(root, encoding='unicode')
```

## Workflow

1. Parse LLM output as XML with lxml (lenient parser)
2. Walk the element tree
3. Remove elements not in the allowlist
4. Strip disallowed attributes from remaining elements
5. Validate structural constraints (path syntax, attribute formats)
6. Serialize cleaned tree back to string

## Key Decisions

- **lxml over regex**: Tree-walking is robust to nesting; regex breaks on complex markup
- **Fail gracefully**: If parsing fails entirely, return a safe default
- **Path validation**: SVG paths with invalid `d` data crash renderers — validate or remove
- **Generalizes to**: HTML sanitization (XSS prevention), XML config validation

## References

- [Drawing with LLMs - Getting Started with Gemma 2](https://www.kaggle.com/code/ryanholbrook/drawing-with-llms-getting-started-with-gemma-2)
