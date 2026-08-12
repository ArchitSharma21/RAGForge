# Resume bullets - v2.0 final

## Recommended final bullet

- Built **RAGForge**, a Dockerized FastAPI/Gradio agentic RAG system with schema-constrained semantic routing, hybrid + hierarchical retrieval, analytical document/table synthesis, read-only Text2SQL, conditional web research, adaptive context compression, cited answers, and a benchmark covering retrieval, citations, robustness, scale stress, latency and release-readiness gates.

## Measured supporting result

- Built a transparent regression suite for retrieval, routing, citation grounding, Text2SQL, robustness, latency, context selection, and synthetic scale stress; used its ablations to drive runtime policies such as small-corpus reranker skipping and adaptive generation context.

- Extended RAGForge with an **Insight Synthesis** path that semantically distinguishes collection overviews from trend/pattern analysis, combines source-balanced document evidence with deterministic DuckDB descriptive context, and grounds structured claims with `[T#]` citations.
- Evolved the evaluation suite into a **hard-mode robustness benchmark** covering paraphrase, distractors, missing answers, multi-hop reasoning, local freshness semantics, analytical synthesis, structured filtering and prompt-injection checks; added hard-mode quality gates.
- Added **chunk-level reranker ablations** and an opt-in Fast/Balanced/Agentic profile benchmark, quantifying evidence quality, latency, LLM-call estimates, correction use and reranker tradeoffs rather than assuming more agentic steps are always better.
- Added **node-level latency observability and timestamped evaluation history/deltas**, turning saved benchmark runs into an inspectable evaluation-driven engineering loop.
# Resume-ready bullets

- Built **RAGForge**, an agentic RAG system with schema-constrained semantic query planning, task-aware routing, dual chunk/source-profile indexes, hierarchical and source-balanced global retrieval, hybrid dense/BM25 search, RRF and cross-encoder reranking.
- Designed a **corrective retrieval loop** that diagnoses weak evidence, rewrites retrieval plans and retries document retrieval before conditionally invoking the web; separated web permission from semantic web relevance to reduce unnecessary external search on private/session-local questions.
- Implemented **task-aware evidence grading** using semantic/lexical relevance, retriever agreement and source coverage, plus Agentic Self-RAG verification, independent document/web query decomposition, HyDE/multi-query expansion and auditable `[D#]/[W#]` citations.
- Added an **Ask-the-Web** research path with parallel search/fetch, extraction and reranking, and an isolated **DuckDB Text2SQL** path for CSV/XLSX analytics with read-only SQL validation.
- Shipped a **Docker Hugging Face Space** with FastAPI + Gradio, per-session isolation, secure ZIP ingestion, OCR fallback, prompt-injection/SSRF defenses, TTL caching, rate limits, Prometheus metrics and non-root writable runtime/cache paths.
- Built a transparent multi-layer RAG benchmark measuring source Hit@1/Recall@5/MRR/AP/nDCG, duplicate-source rate, answer accuracy, citation validity/coverage, planner/web-policy accuracy, Text2SQL, abstention, cache-bypassed latency and reranker ablations, with quality-gated grades and calibrated Gemini judging.
- Made evaluation **incremental and quota-aware** by caching Quick/Standard/Deep reports per workspace, comparing runs in-app, reusing compatible Standard deterministic results for judge-only Deep evaluation, and exposing saved reports through FastAPI.
- Converted benchmark findings into runtime optimization with **adaptive reranking**, skipping a multi-second CPU cross-encoder on easy/small-corpus paths when repeated ablations showed no source-ranking gain while retaining reranking for harder/larger tasks.
- Added **typed Text2SQL evaluation** and zero-call evidence-aware citation repair, separating SQL correctness from rendering quirks and improving citation completeness without additional LLM requests.

- Hardened public RAG lifecycle UX with browser-persistent workspace IDs, lazy demo re-indexing after ephemeral Space restarts, explicit empty-corpus/insufficient-evidence abstention, staged ingestion progress, and inspectable workspace/evidence traces.

- Instrumented LangGraph traces with node latency, correction/web flags and estimated LLM-call counts; added visible run-state/duplicate-click protection for chat and evaluation, and an interactive Architecture + API view with live workspace metadata, endpoint reference and copy-ready curl examples.

- Added **quota-aware evaluation infrastructure** with a rolling per-model Gemini request budget, provider-guided 429 backoff, pacing-vs-service latency separation, request telemetry, sampled LLM judging, and one-call Text2SQL component checks for reliable free-tier benchmark runs.


## v1.5.1 evaluation UX / reliability angle

- Added JSON-safe evaluation persistence and copy/export tooling for all benchmark result tables, keeping saved Quick/Standard/Deep reports directly comparable without rerunning model calls.
- Hardened zero-call citation post-processing with grouped-citation normalization, duplicate-tail cleanup and conservative preamble skipping to improve citation usability without increasing inference cost.

- Built an evaluation-driven RAG quality loop (v1.7) that corrected Markdown-aware citation scoring and missing-answer evaluation, added run-level cache provenance, and converted source/chunk ablations plus Fast/Balanced/Agentic benchmarks into adaptive reranker/profile policy diagnostics.

## v1.8 evidence-driven efficiency bullet

- Converted RAG evaluation signals into a context-budget policy: added conservative focused-query pruning and prompt-budget telemetry, then benchmarked full top-k vs pruned context on source precision/recall and estimated token load without extra LLM calls; broad analytical and multi-source tasks retain full evidence breadth.


## v1.9 scale / efficiency bullets

- Generalized evaluation-driven context pruning into a **corpus-aware retrieval budget**: dynamically expanded candidate depth with corpus scale, selected 2-5 evidence chunks from ranking confidence/ambiguity, and applied zero-LLM sentence compression while preserving broad multi-source synthesis paths.
- Built **zero-Gemini scale and context ablations** that stress the real hybrid retriever at roughly 1x/5x/20x long-document distractor scale, compare full/fixed/adaptive context policies, validate answer-signal retention, and gate acceptance checks on quality/robustness thresholds.
- Added **prompt-economics and operational observability** including estimated generation input/output tokens, cited-evidence utilization, pace-corrected node latency, workspace capacity/index-memory diagnostics and a release-readiness checklist exposed in Gradio and FastAPI.
