---
name: nlp-whoosh-fulltext-search-index
description: >
  Builds a Whoosh full-text search index over documents and queries it with boolean operators, field scoping, and proximity matching.
---
# Whoosh Full-Text Search Index

## Overview

Whoosh is a pure-Python full-text search library (no external dependencies) that supports boolean queries, field-scoped search, wildcards, and proximity operators. It's ideal for Kaggle competitions and prototyping where Elasticsearch is unavailable. Build an index over document fields (title, abstract, classification codes), then query with structured boolean expressions. Supports BM25 scoring out of the box.

## Quick Start

```python
import os
from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import MultifieldParser, OrGroup

# Define schema
schema = Schema(
    doc_id=ID(stored=True, unique=True),
    title=TEXT(stored=True),
    abstract=TEXT,
    cpc=TEXT,
)

# Build index
ix_dir = "my_index"
os.makedirs(ix_dir, exist_ok=True)
ix = index.create_in(ix_dir, schema)
writer = ix.writer()
for doc in documents:
    writer.add_document(
        doc_id=doc['id'], title=doc['title'],
        abstract=doc['abstract'], cpc=doc['cpc_codes']
    )
writer.commit()

# Query with field scoping and boolean operators
ix = index.open_dir(ix_dir)
searcher = ix.searcher()
qp = MultifieldParser(["title", "abstract", "cpc"], schema, group=OrGroup)

query = qp.parse("title:neural OR abstract:classification OR cpc:G06F")
results = searcher.search(query, limit=50)
hits = [r['doc_id'] for r in results]
```

## Workflow

1. Define a schema with field types (TEXT for searchable, ID for stored keys)
2. Create index and add documents via writer
3. Open index and create a searcher
4. Build queries with field scoping (`field:term`), boolean ops (`OR`, `AND`), wildcards (`term*`)
5. Retrieve ranked results by BM25 score

## Key Decisions

- **Schema design**: Use TEXT for searchable fields, ID for unique keys, KEYWORD for tags
- **Field scoping**: `title:word` searches only the title field — reduces false positives
- **Proximity**: `ADJ5` (within 5 positions) for phrase-like matching, but doesn't work with wildcards
- **Performance**: Whoosh is single-threaded — for 100k+ docs, consider batching queries

## References

- [Basic Whoosh Search Demo](https://www.kaggle.com/code/sohier/basic-whoosh-search-demo)
- [USPTO - Annealing [LB = 0.31]](https://www.kaggle.com/code/andrey67/uspto-annealing-lb-0-31)
