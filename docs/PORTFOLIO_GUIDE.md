# Demo guide

This page is a short walkthrough for showing RAGForge without turning the demo into a feature checklist.

## Suggested demo

Use the bundled corpus with `Auto` + `Balanced`.

### Focused document question

```text
What is the Sev-1 acknowledgement target?
```

Show the answer, the returned source, and the Pipeline Inspector. The useful point is that retrieval can start broad enough for safety while the generation context is reduced when the evidence is clear.

### Corpus overview

```text
What is the collection about?
```

Show that the result covers the different source files rather than being dominated by the long PDF. The support table should appear as structured evidence rather than being treated only as raw CSV text.

### Cross-source analysis

```text
What exactly does this collection reveal? Identify important trends and caveats.
```

This demonstrates the analytical path, which combines document evidence with deterministic table summaries before generation.

### Structured data

```text
Which support tier has the shortest first-response SLA?
```

Show that the planner sends the question to the read-only SQL path and that the query/result are inspectable.

### Evaluation

Open a Standard result and show one or two tables rather than the whole report. The most useful examples are:

- retrieval vs reranking,
- full vs fixed vs adaptive context budgets,
- sentence-compression signal retention,
- node latency, and
- synthetic scale stress.

The main point is that evaluation changed runtime policy. Components are not enabled just because they are common RAG techniques.

## Short architecture explanation

RAGForge first decides whether a question belongs to documents, structured data, the web, or a mixed path. Document retrieval can be direct semantic search, source-balanced global search, source-first hierarchical search, or analytical document-plus-table retrieval. Dense and lexical results are fused, then runtime policies decide retrieval depth, reranking, and how much evidence reaches generation. The answer returns its sources and an execution trace so those choices can be inspected.

## Engineering tradeoffs worth discussing

### Reranking

The project includes a cross-encoder, but the small demo benchmark did not show a ranking improvement that justified its multi-second cost. The runtime therefore skips it for the measured small-corpus case and keeps it available for larger/harder workloads.

### Context size

Source recall was strong while precision was modest. Instead of globally lowering retrieval depth, RAGForge separates retrieval from generation context and uses an adaptive evidence budget for focused questions.

### Evaluation cost

A full benchmark can hit free-tier RPM limits. Evaluation uses a rolling request pacer, stores completed reports, and lets Deep reuse a compatible Standard baseline.

### Evaluator correctness

The benchmark itself is tested and versioned. During development it exposed bugs in AP calculation, citation parsing, missing-information scoring, and numeric answer matching. A benchmark result is only useful if the measurement code is also trustworthy.

## Limitations to mention

- state is ephemeral on a standard Hugging Face Space
- there is no durable multi-tenant storage layer
- prompt-injection filtering is heuristic
- the larger retrieval test uses synthetic distractors
- trace token counts are estimates rather than billing records
- production deployment would need stronger tenant isolation, durable storage, and external infrastructure

## Resume description

> Built a FastAPI/Gradio RAG system that routes questions across documents, structured data, and web search, combines dense and lexical retrieval, returns source-linked answers, and includes regression tests for retrieval, routing, SQL, robustness, latency, and context-selection policies.
