# Migration to RAGForge v1.5.2

v1.5.2 is a narrow hotfix over v1.5.1.

## Fixed

Quick, Standard and Deep evaluation could fail while processing generated answers with:

```text
NameError: name 'repair_missing_citations' is not defined
```

`src/ragforge/pipeline.py` called citation helpers moved into `src/ragforge/citations.py`, but the v1.5.1 package omitted the explicit module import. v1.5.2 restores:

```python
from .citations import normalize_citation_syntax, repair_missing_citations
```

## Compatibility

- No dependency changes.
- No corpus/index format changes.
- No retrieval-policy changes.
- No evaluation-scoring changes.
- The demo benchmark remains version `1.5.1`, so saved reports are semantically compatible while the same server-side workspace exists.

## Regression coverage

The test suite now checks the import wiring without optional dependencies and invokes the citation-repair pipeline helper when LangGraph is available.
