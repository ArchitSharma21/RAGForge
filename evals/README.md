# RAGForge demo benchmark

The benchmark covers retrieval, routing, context policies, structured data, robustness, and scale stress. The multi-source Hard Mode comparison runs through the recommended Auto + Balanced path: `hard_multihop_time_compare` now runs under `Auto + Balanced` and explicitly expects `documents -> comparison -> hierarchical`. This matches the recommended semantic runtime instead of using Fast mode as the reference for a multi-source comparison.

## Local ablations and scale stress

Standard/Deep add three zero-Gemini evaluation surfaces: a three-way full-vs-fixed-vs-adaptive context budget, focused sentence-compression signal retention, and a 1x/5x/20x synthetic long-document distractor stress harness. These are regression/engineering tests for the bundled demo, not claims of universal large-corpus performance. The release-readiness table applies explicit thresholds to the underlying metrics and marks missing Standard/Deep stress evidence as incomplete rather than silently passing it.

`demo_benchmark.json` is a small, inspectable hard-mode benchmark tied to the bundled demo corpus. It intentionally separates:

- focused QA labels - expected answer terms and relevant source files,
- corpus-overview behavior - breadth, source coverage and unnecessary web use,
- semantic planner behavior - expected route, task, retrieval strategy and whether web access is appropriate,
- Text2SQL component behavior - validated read-only SQL generation/execution and typed expected scalar values when labeled,
- lifecycle abstention - explicit missing-resource cases that should use zero model calls.

The benchmark is not meant to claim general RAG performance. It is a regression and architecture-validation suite for this demo corpus.

## Standard vs Deep evaluation

**Standard** uses deterministic labels wherever possible: answer-key terms, source Hit@1/Recall@K/MRR/AP/nDCG, citation validity/coverage, route/task/strategy accuracy, web-use precision/recall, overview source coverage, Text2SQL result checks and latency/trace efficiency.

**Deep** adds an auxiliary Gemini judge for a representative labeled subset spanning ordinary QA, NIST, cross-document synthesis and corpus overview. Judge scores are reported separately from deterministic metrics because LLM-as-judge evaluation is itself probabilistic.

## Quota-safe execution

v1.4.1 introduced, and v1.5 retains, a default 12 RPM evaluation request budget. The process-local request ledger accounts for recent interactive requests made with the same key/model, and surfaced 429 responses honor provider retry guidance before bounded retries. Deliberate pacing time is reported separately from service latency.

The Text2SQL component test uses one model call per case because SQL routing is already evaluated independently in the semantic-planner suite. This avoids duplicating route and answer-generation calls solely for the benchmark.

## v1.5 evaluation reuse

Completed Quick, Standard and Deep reports can be saved per workspace and compared without rerunning. Reuse requires matching corpus version, benchmark version and model. Older/stale reports remain viewable but are not silently reused after the corpus changes.

Deep can be incremental: when a compatible Standard report exists, v1.5 reuses the deterministic Standard rows and exact stored answer/evidence artifacts, then issues only the sampled calibrated judge calls. This reduces free-tier request pressure while making Standard-vs-Deep comparison deterministic.

Text2SQL cases with labeled scalar outputs use typed comparison (`bool`, numeric or text) rather than relying on Markdown rendering. The explicit reranker ablation remains in Standard/Deep even though the runtime can now skip the cross-encoder adaptively for easy/small-corpus paths.


## v1.6 hard-mode cases

The `hard_mode_cases` section intentionally targets failure modes that the original easy benchmark did not stress: paraphrase, distractors, missing information, multi-hop comparison, analytical synthesis, filtered structured queries, local-current wording and prompt-injection detection.

Selected QA cases also define `chunk_must_contain`. These labels are used only for chunk-level reranker Hit@1/MRR; unlabeled cases are excluded from those averages.

The optional profile benchmark is not stored as a separate question set: it reuses a focused QA case and a cross-document case across Fast, Balanced and Agentic so the comparison is controlled.

## v1.7 scoring corrections

Citation coverage is Markdown-aware, missing-answer cases accept natural grounded-absence language, and global overview table citations are evaluated against actual `[T#]` source records. Reports also include run/server provenance and optional aggregated profile-policy summaries. Benchmark version `1.7` intentionally prevents automatic reuse of v1.6 reports under the changed scoring semantics.

## v1.8 context-budget ablation

Standard/Deep now add a deterministic `context_budget_ablation` table. It compares the original six-chunk focused context with the adaptive three-chunk safety floor and reports source Precision@5, Recall@5, Hit@1, MRR, median chunks/sources/chars, estimated tokens and reduction percentage. The ablation issues no Gemini requests.

The missing-answer Hard Mode row also exposes `missing_answer_match` and `grounded_absence`. Either calibrated signal can satisfy the missing-information decision, avoiding false failures caused by one exact wording.
