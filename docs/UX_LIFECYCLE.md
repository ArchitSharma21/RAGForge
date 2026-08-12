# UX and lifecycle - v2.0 final

## Final product polish

v2.0 gives the UI a compact product hero, verified-baseline cards, clearer Chat/Evaluation hierarchy, recommended-profile guidance, evaluation metric cards, a cleaned runtime snapshot, and a final project-status footer. The underlying session/lazy-demo/indexing/query/evaluation lifecycle remains the same.

## v1.9 runtime health and capacity visibility

The Architecture + API runtime snapshot now reports corpus scale, configured chunk-capacity utilization, approximate vector-index memory, session age/idle time, TTL, index readiness and archived evaluation count. This does not make the Hugging Face container persistent; it makes the limits of the current ephemeral workspace visible.

Focused query inspection also distinguishes retrieval depth, adaptive context budgeting and sentence compression. These stages are displayed separately so a smaller generation prompt cannot be mistaken for lower retrieval recall.

## Goals

The public demo should remain understandable even when indexing takes several seconds, a browser is refreshed, or the Hugging Face container restarts. Lifecycle state is handled explicitly rather than surfacing as a misleading retrieval failure.

## Corpus build states

Manual indexing immediately changes the UI from `Index corpus` to `Building corpus...` and disables the button. The ingestion callback reports these coarse stages through `gr.Progress`:

1. preparing inputs / safe ZIP expansion
2. parsing supported files
3. chunking
4. building the dense + BM25 chunk index
5. building source profiles
6. building the source-profile index
7. corpus ready

The final corpus card reports source count, document units, chunks, source profiles and table names.

## Browser state

The UI stores one opaque session ID in `gr.BrowserState`. No document text, embeddings, API key, SQL data or chat history is written into browser local storage.

- Normal refresh while the same Space process is alive: reconnect to the existing workspace.
- Session TTL expiry: a fresh workspace is created and the UI explains that the old session expired.
- Space/container restart: the browser ID may remain, but the server-side workspace is gone. The UI creates a fresh server workspace.

## Lazy demo initialization

When `Use bundled demo files` is enabled and the current workspace is empty, the first non-Web query indexes the bundled demo corpus before running the RAG pipeline. This is intentionally UI-specific convenience behavior. The REST API does not silently ingest demo files.

Custom uploads are never reconstructed implicitly after a container restart; users must upload/index them again.

## Pipeline preflight and abstention

The LangGraph route node checks whether the selected knowledge path actually has local data:

- document route + zero chunks -> terminal `abstain`
- SQL route + zero tables -> terminal `abstain`
- hybrid route + no local corpus -> use web only when external evidence is semantically relevant and allowed; otherwise abstain

After corrective retrieval, a corpus-only task with insufficient local evidence also goes to `abstain` instead of generation -> verification -> revision.

## Inspector fields

Each query trace includes a `workspace` block with status, version, source/chunk/profile/table counts. The UI also renders a compact summary of the semantic plan, evidence score/coverage and execution path, while retaining raw JSON for debugging.


## Long-running UI feedback

Manual ingestion keeps one explicit `gr.Progress()` surface and suppresses Gradio's default full overlay. Query and evaluation flows use dedicated status lines instead of a second progress overlay:

- Ask immediately changes to `Processing...`, disables repeat submission and shows a processing message.
- Evaluation immediately changes to `Running <level> evaluation...`, disables repeat clicks and explains that the tab should remain open.
- Both controls restore their normal interactive state on success or error.
- This design prevents the silent-wait problem without reintroducing the overlapping progress UI fixed in v1.3.

## Saved evaluation lifecycle - v1.5

Quick, Standard and Deep evaluation reports are cached inside the current workspace, separately from the normal query response cache. The Evaluation tab can load any saved depth and compare saved runs without another benchmark execution.

Each saved report records the model, benchmark version and workspace/corpus version. If the corpus changes, the old report remains viewable but is marked stale and is not eligible for automatic reuse. A normal browser refresh can recover saved runs while the same server workspace lives. A Hugging Face container restart still removes the ephemeral workspace and therefore its saved evaluations.

With `Reuse saved evaluation` enabled:

- rerunning the same compatible depth returns the saved report with zero Gemini calls;
- Deep reuses a compatible Standard deterministic baseline and adds only the sampled judge layer;
- disabling reuse forces a fresh cache-bypassed benchmark run.


## Saved evaluation viewing and export

Switching Quick/Standard/Deep is a read-only operation when a saved report exists. v1.5.1
normalizes the report before rendering it as formatted JSON so changing depths does not degrade the
raw report into a Python/Pydantic representation.

The Evaluation tab can also prepare any result table as CSV, TSV or Markdown. The copy-ready text
and download file are derived from the selected saved run; no Gemini request is made.


## v1.6 evaluation history and analytical UI

Saving an evaluation still updates the latest Quick/Standard/Deep cache, but v1.6 also writes a timestamped historical copy inside the workspace. History is read-only and consumes no model quota. It is ephemeral with the Space container, just like the corpus.

The query inspector adds a text-based node-latency waterfall. This uses existing LangGraph trace timings and introduces no plotting dependency or extra model call. Structured evidence produced by analytical synthesis appears in the source panel as `[T#]` table cards with row count, schema and a bounded preview.

## v1.7 source, trace and saved-run UX

Source snippets are rendered as escaped plain text inside uniform source cards. Source-controlled Markdown therefore cannot change font size or create headings in the Sources accordion.

The Pipeline Inspector latency waterfall is now a proportional graphical bar display rather than a row of `#` characters. The same section exposes the adaptive reranker decision and whether the response is a grounded absence answer.

Saved evaluation switching is bound to user input rather than generic change events. A fresh benchmark can update the selected depth without immediately reloading itself and overwriting the completion message. Fresh and reused statuses include the saved run ID; evaluation metadata also records the server boot that produced the report.

## v1.8 context-budget observability

Pipeline Inspector now exposes the focused context-budget decision directly after the reranker decision. For eligible fact lookups it shows chunks and estimated tokens before/after pruning plus reduction percentage. Broad tasks show a skip reason rather than a misleading zero.

The Evaluation tab adds a **Context budget** table and export option. This ablation is deterministic and uses no additional Gemini calls, so users can inspect context-efficiency tradeoffs without increasing free-tier request pressure.