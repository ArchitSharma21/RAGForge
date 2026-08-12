# Evaluation

RAGForge includes a small, transparent regression suite in `evals/demo_benchmark.json`. It is designed to diagnose parts of the system separately rather than reduce everything to one headline score.

## Evaluation levels

### Quick

Quick is a deployment smoke test. It runs a subset of:

- focused QA,
- query planning,
- corpus overview,
- Text2SQL,
- robustness cases, and
- empty-workspace abstention.

Quick intentionally skips the larger context-budget, sentence-compression, reranker, and scale-stress ablations. The UI reports those sections as **not run** rather than turning skipped work into a readiness result.

### Standard

Standard runs the full deterministic suite:

- focused QA answer checks,
- source-level retrieval metrics,
- planner route / task / strategy labels,
- corpus overview behavior,
- Text2SQL computed-value checks,
- abstention,
- hard-mode robustness,
- RRF vs reranker ablation,
- full vs fixed vs adaptive context budgets,
- focused sentence-compression checks,
- node latency, and
- synthetic distractor scale stress.

### Deep

Deep reuses a compatible Standard result when possible and adds a small LLM-judge sample. The judge covers faithfulness, relevance, completeness, and citation support, but deterministic citation failures cannot be overridden by the judge.

## Answer matching

Focused QA uses labeled answer phrases from the benchmark file. Matching is boundary-aware: numeric or alphanumeric labels cannot pass by appearing inside a different token. For example, a `5 min` answer key does not match `15 minutes`.

This rule was added after a real v2.0 evaluation exposed exactly that false positive.

## Retrieval metrics

Source-level retrieval deduplicates repeated chunks from the same file before computing:

- Precision@5,
- Recall@5,
- Hit@1,
- MRR,
- AP@5, and
- nDCG@5.

Selected cases also include chunk-content labels for chunk Hit@1 and chunk MRR in the reranker ablation.

A source-localization question may legitimately use either direct semantic retrieval or source-first hierarchical retrieval; benchmark cases can declare more than one accepted strategy when both are valid.

## Citation checks

Citation validity asks whether cited source IDs were actually returned with the answer. Citation coverage checks whether substantive factual units have citations. The parser is Markdown-aware so headings and list introductions are not counted as unsupported factual claims, while factual list items are.

Structured table evidence and web evidence are validated against their returned source records in the same way as document evidence.

## Text2SQL

SQL generation is evaluated independently from natural-language answer generation. The benchmark:

1. generates validated read-only SQL,
2. executes it in DuckDB, and
3. compares the computed scalar result using typed boolean, numeric, or text matching.

## Context-budget ablation

Standard compares:

- **Full top-k** - all retrieved candidates used as generation context;
- **Fixed 3-chunk budget** - a deliberately tight focused baseline; and
- **Adaptive budget** - a 2-5 chunk focused budget chosen from retrieval confidence, score separation, source ambiguity, and corpus scale.

The purpose is to check whether model input can be reduced without losing the labeled source.

## Sentence-compression ablation

Focused evidence can be reduced to query-relevant sentences after context budgeting. The ablation checks whether labeled answer evidence remains present and records the resulting context-size change. Original source cards are never rewritten.

## Synthetic scale stress

The stress harness clones long-document distractor chunks and reuses their existing vectors to exercise the real Qdrant + BM25 + RRF path at larger index sizes without extra Gemini or embedding calls.

It is a retrieval regression test, not a replacement for evaluation on a real large corpus.

## Latency and request pacing

Evaluation bypasses the response cache. Service latency excludes deliberate quota-pacing sleep, while wall time and pacing wait are reported separately.

All Gemini calls in a run can share a rolling request pacer. Saved Quick, Standard, and Deep reports can be reopened without spending new requests, and Deep can reuse Standard as its deterministic baseline.

## Acceptance checks

Standard and Deep include a threshold checklist for the major benchmark subsystems. It is an engineering convenience, not a product-quality certification and not a substitute for the underlying tables.

## Benchmark scope

The bundled corpus is deliberately small. High percentages on this suite mean the current implementation passes these labeled regression cases. They should not be read as general accuracy estimates for arbitrary documents, industries, or enterprise-scale deployments.
