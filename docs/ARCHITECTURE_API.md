# Architecture and API - v2.0 final

## Final system shape

v2.0 freezes the feature architecture around semantic routing, hybrid/source-balanced/hierarchical/analytical retrieval, adaptive context budgeting, focused evidence compression, Text2SQL, conditional web research, grounded generation, transparent traces, and component-level evaluation. The final release focuses on product presentation, benchmark alignment, documentation, and release verification rather than adding another retrieval subsystem.

## v1.9 adaptive scale and context pipeline

Focused retrieval now separates **retrieval depth** from **generation context budget**. Candidate depth grows with corpus scale (6/8/10/12 minimum candidates across small/medium/large/very-large workspaces), then the context policy chooses a 2-5 chunk budget from retrieval confidence, score gap and source ambiguity. A final deterministic sentence-compression stage can trim the generation copy of focused evidence while preserving original source cards. Broad overview, insight, comparison and cross-document tasks bypass focused compression/budget narrowing.

The retrieve/generate trace includes corpus scale, effective top-k, retrieval confidence, score gap, context target, pre/post context tokens, compression reduction and prompt/output token estimates. `GET /api/v1/session/{session_id}/diagnostics` exposes workspace capacity/TTL, index readiness, estimated vector-memory footprint and evaluation-history state.

Standard/Deep additionally run a local synthetic scale-stress harness and a release-readiness checklist. None of the new v1.9 ablations require extra Gemini calls.

## v1.8 context-budget policy

Focused local lookups now pass through a deterministic context-budget policy after ranking and before evidence grading/generation. The policy is deliberately narrow: only `fact_lookup`/`followup` tasks using semantic or hierarchical local retrieval are eligible, and the context retains a three-chunk safety floor. Global, analytical, comparison and cross-document tasks bypass pruning.

The `retrieve` trace exposes `context_pruning_used`, reason, chunks/sources before and after, estimated tokens and reduction percentage. The `generate` trace exposes whether the corpus manifest was included plus evidence-context and prompt-size estimates. For focused fact lookups the manifest is omitted because semantic routing has already established corpus scope; broad/mixed tasks retain it.

Standard/Deep evaluation include a zero-Gemini `context_budget_ablation` comparing full top-k with the focused budget before treating any token reduction as a quality win.


## Runtime architecture

RAGForge uses one FastAPI application with a mounted Gradio UI. Each browser/API session maps to an isolated in-process `Workspace` containing document units, chunk/source indexes, DuckDB tables, history and corpus version.

### LangGraph path

```text
request
  -> guard
  -> semantic route
  -> workspace preflight
  -> plan
  -> semantic/global/hierarchical/analytical/table/web retrieval
  -> adaptive reranker policy (skip or cross-encoder)
  -> evidence grade
  -> optional correction + retry
  -> conditional web augmentation
  -> grounded generation
  -> verification / bounded revision
  -> cited response or abstention
```

The Architecture + API tab exposes the responsibilities of each graph node in a live DataFrame.

The adaptive reranker decision is recorded in the retrieval trace as `reranker_used` and `reranker_reason`. Standard/Deep evaluation still runs an explicit on/off ablation so the runtime decision remains measurable.

## Live workspace snapshot

`Refresh runtime view` reports:

- app version,
- workspace status/version,
- source count,
- chunk count,
- source-profile count,
- table count,
- saved evaluation depths,
- configured generation/embedding/reranker/search models.

It also generates curl examples using the current browser workspace ID and reports saved evaluation inventory through workspace stats.

## REST surface

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | health check |
| GET | `/api/v1/info` | service/model/features metadata |
| POST | `/api/v1/session` | create session |
| GET | `/api/v1/session/{session_id}` | inspect workspace status |
| POST | `/api/v1/ingest` | multipart document ingestion |
| POST | `/api/v1/query` | execute RAG query |
| POST | `/api/v1/evaluate/demo` | Quick/Standard/Deep benchmark |
| GET | `/api/v1/evaluation/benchmark` | benchmark metadata/counts |
| GET | `/api/v1/evaluation/saved/{session_id}` | list saved Quick/Standard/Deep runs |
| GET | `/api/v1/evaluation/saved/{session_id}/{level}` | retrieve one saved evaluation report |
| GET | `/api/v1/evaluation/history/{session_id}` | list timestamped evaluation history and deltas |
| GET | `/docs` | Swagger UI |
| GET | `/openapi.json` | OpenAPI schema |
| GET | `/metrics` | Prometheus metrics |

When `APP_API_TOKEN` is set, protected endpoints require a Bearer token.

## Query example

```bash
curl -X POST http://localhost:7860/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "SESSION_ID",
    "query": "What is the collection about?",
    "config": {
      "mode": "Auto",
      "profile": "Balanced",
      "model": "gemini-3.5-flash-lite"
    }
  }'
```

## Evaluation example

```bash
curl -X POST http://localhost:7860/api/v1/evaluate/demo \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "SESSION_ID",
    "level": "Standard",
    "model": "gemini-3.5-flash-lite",
    "target_rpm": 12,
    "reuse_saved": true,
    "include_profile_benchmark": false
  }'
```

When `reuse_saved=true`, a compatible saved report can be returned with zero Gemini requests. For Deep, a compatible saved Standard report can be reused as the deterministic baseline so only the sampled judge layer is added.

Saved evaluations can be inspected without rerunning:

```bash
curl http://localhost:7860/api/v1/evaluation/saved/SESSION_ID
curl http://localhost:7860/api/v1/evaluation/saved/SESSION_ID/Standard
```

## Storage lifecycle

Standard Hugging Face Space disk is ephemeral for this deployment design. Browser state stores only the opaque workspace ID. A normal refresh can reconnect while the process lives; a container restart removes in-memory indexes and custom uploads must be re-indexed. Bundled demo data can be lazily rebuilt.

Evaluation reports are stored inside the same ephemeral workspace. They survive a normal browser refresh while the workspace/container lives, but are not durable production storage. Reports include model/benchmark/corpus-version metadata so stale runs are visible rather than silently reused after corpus changes.

## Evaluation quota controls

`POST /api/v1/evaluate/demo` accepts `target_rpm`. The UI defaults to 12 RPM for quota-safe portfolio/free-tier runs. The benchmark uses one shared rolling request budget across planner, generation, Text2SQL and Deep-judge calls, and the raw report exposes request/pacing telemetry.


## Evaluation report portability in v1.5.1

Saved Quick, Standard and Deep reports are converted to plain JSON before persistence and API
return. This keeps `GET /api/v1/evaluation/saved/{session_id}/{level}` structurally identical to a
fresh evaluation response and avoids UI-framework wrapper representations.

The Gradio Evaluation tab also exposes a table export panel. This is a UI convenience rather than a
new network API: it materializes CSV/TSV/Markdown files inside the current ephemeral workspace.


## v1.6 analytical evidence path

`insight_synthesis` requests use the `analytical` strategy. Source-balanced original document chunks remain `[D#]` evidence. DuckDB contributes deterministic schema, bounded rows and descriptive signals as `[T#]` evidence. One grounded generation call synthesizes patterns, quantitative signals, contrasts and caveats. The table context is not an LLM-generated summary, so no additional API request is spent preparing it.

## v1.6 evaluation observability

The UI exposes Hard Mode, optional profile comparison, node-latency summaries and timestamped evaluation history. `GET /api/v1/evaluation/history/{session_id}` exposes the same archived-run metadata to API clients. The normal latest-run endpoints remain unchanged.

## v1.7 runtime provenance and calibrated absence

Evaluation cache metadata includes a short run ID and server-boot ID. These fields make it possible to distinguish a fresh benchmark from a saved report in the UI/API without relying on ambiguous status text. `GET /api/v1/evaluation/saved/{session_id}` includes this provenance in its inventory.

The verify path now recognizes a grounded absence answer: an evidence-cited statement that the requested fact is not present in the selected sources. This state skips the normal low-confidence revise branch, preventing a second generation call whose only purpose would be to restate the same absence.

For corpus overviews, structured tables are surfaced as deterministic `[T#]` evidence alongside the source-balanced document set. Analytical synthesis continues to use the same table evidence path.