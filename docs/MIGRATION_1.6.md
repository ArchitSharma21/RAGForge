# Migration to RAGForge v1.6

v1.6 is a capability and evaluation release over v1.5.2. Runtime dependency pins are unchanged.

## Main changes

- new `insight_synthesis` task type and `analytical` retrieval strategy;
- deterministic DuckDB analytical context with `[T#]` table citations;
- semantic high-threshold fallback for zero-call citation attribution;
- benchmark version `1.6` with Hard Mode robustness cases;
- source + chunk reranker ablation;
- optional Fast/Balanced/Agentic profile benchmark;
- node-latency waterfall and aggregate node timing table;
- timestamped evaluation history + API endpoint;
- updated README, architecture, evaluation, planning, feature matrix and resume bullets.

## Saved evaluation compatibility

The benchmark version changes from `1.5.1` to `1.6`, so old reports remain historical but are not eligible for automatic benchmark reuse under the new scoring/case semantics.

## API

`EvaluationRequest` gains `include_profile_benchmark` (default `false`). A new `GET /api/v1/evaluation/history/{session_id}` endpoint returns archived run metadata and deltas. Existing endpoints remain backward-compatible.
