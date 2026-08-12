# Feature matrix - v2.0 final

The table below describes the final planned portfolio release. Future changes should be driven by real deployment/corpus requirements rather than feature-count growth.

| Capability | RAGForge implementation | Why it matters |
|---|---|---|
| Semantic query planning | schema-constrained Gemini `QueryPlan` in Balanced/Agentic | resolves intent before retrieval instead of relying on freshness/SQL keyword regexes |
| Knowledge-scope classification | corpus / external / mixed / structured-data | prevents session-local phrases such as “current corpus” from being mistaken for web freshness |
| Task classification | fact / overview / cross-doc / comparison / aggregation / insight synthesis / follow-up | lets evidence requirements depend on the task and distinguishes broad description from analytical pattern-finding |
| Independent retrieval queries | separate document and web queries | search engines and private corpora often need different formulations |
| Corpus manifest | source metadata + deterministic representative excerpts | gives the planner grounded knowledge of what is actually indexed |
| Source-profile index | one retrieval-only profile per source | enables document-level routing before chunk-level retrieval |
| Hierarchical retrieval | source profile → selected source(s) → source-scoped chunk retrieval | improves broad/source-specific questions and avoids unrelated chunk competition |
| Global retrieval | source-balanced evidence selection | corpus overviews represent distinct files instead of the longest document |
| Analytical retrieval | source-balanced document evidence + deterministic DuckDB table context | supports grounded trend/insight synthesis across mixed local evidence without extra table-summary LLM calls |
| Dense retrieval | FastEmbed BGE-small + Qdrant local | semantic recall without hosted embedding cost |
| Source-scoped dense retrieval | cached normalized embedding matrix | efficient hierarchical search inside selected sources without rebuilding stores |
| Sparse retrieval | BM25 | exact terms, identifiers, error codes and names |
| Hybrid fusion | Reciprocal Rank Fusion | robust combination of lexical + semantic rankings |
| Reranking | local MiniLM cross-encoder behind an adaptive profile/task/corpus policy | retains second-stage precision capability while avoiding measured multi-second overhead on easy/small-corpus paths |
| Relevance display | rank + bounded hybrid retrieval signal | avoids presenting uncalibrated cross-encoder logits as probabilities |
| Chunking | sentence-aware overlap + optional semantic breakpoints | balances context continuity and retrieval granularity |
| Multi-query | planner-generated variants in Balanced/Agentic, bounded by profile | improves recall across alternate wording |
| HyDE | optional Agentic hypothetical passage used only for retrieval | bridges abstract questions to document language |
| Corrective RAG | retrieve → task-aware grade → semantic correction → retrieve again | fixes weak retrieval before reaching for external data |
| Conditional web fallback | web requires both permission and semantic relevance | prevents irrelevant internet pollution of private/session-local questions |
| Evidence grading | relevance + method agreement + task-aware source coverage + optional LLM judge | a global overview and a focused fact lookup should not share one confidence rule |
| Abstention | corpus-only tasks can answer “insufficient evidence” after correction | weak retrieval does not automatically imply the internet is relevant |
| Self-RAG | faithfulness audit + bounded revision | reflection without unbounded recursion |
| Ask-the-Web | independent query fan-out, parallel search/fetch, extraction, rerank, synthesis | fresh-information path separated from private retrieval |
| Native search | Gemini grounding submodel | provider-native search option while keeping main RAG model independent |
| Routing controls | Auto/Documents/Web/Hybrid/Data(SQL) | semantic defaults plus explicit user override |
| Text2SQL | isolated DuckDB + validated read-only SQL | structured questions are computed rather than guessed from text chunks |
| OCR | optional Gemini file transcription | scanned documents/images remain usable |
| Multiformat ingestion | PDF/TXT/MD/DOCX/PPTX/CSV/XLSX/JSON/HTML/code/images/ZIP | realistic enterprise ingestion surface |
| ZIP hardening | traversal/file-count/uncompressed-size/type limits | archive UX without naive extraction risk |
| Citations | `[D#]` documents, `[T#]` tables, `[W#]` web + conservative zero-call lexical/semantic repair | auditable answer grounding across text and structured evidence with no extra citation-fixer request |
| Prompt-injection defense | untrusted-context rules + heuristic scoring/downranking | retrieval is an attack surface |
| SSRF defense | public URL/DNS checks + redirects disabled | web agents must not become internal-network fetchers |
| Session isolation | per-session corpus/index/DuckDB/history/cache version | prevents accidental cross-user context |
| Caching | TTL result cache keyed by session/config/corpus version | latency/quota reduction without stale cross-corpus answers |
| Rate limiting | sliding-window per IP | protects a public shared model key |
| Observability | Prometheus + semantic plan/evidence/correction/node trace + node time/estimated LLM calls/web/correction/reranker/citation-repair flags | makes agent decisions and efficiency inspectable |
| Adaptive retrieval depth | corpus-scale-aware 6/8/10/12 candidate depth | avoids assuming the 90-chunk demo top-k is sufficient for larger uploads |
| Adaptive context budget | focused 2-5 chunk budget from scale, score gap, confidence and source ambiguity | shrinks generation context without collapsing broad or ambiguous evidence needs |
| Focused evidence compression | deterministic query-relevant sentence selection for generator context only | reduces prompt load without mutating source cards or spending another LLM call |
| Scale stress | 1x/5x/20x long-document distractor cloning with precomputed vector reuse | tests retrieval/budget robustness around ~1000 chunks without Gemini or re-embedding cost |
| Workspace diagnostics | corpus scale/capacity, index readiness, TTL/idle, estimated vector memory | makes demo operational limits explicit before they become silent failures |
| Acceptance checks | transparent threshold checklist over quality, safety, efficiency and scale robustness | summarizes ship/no-ship evidence without hiding component metrics |
| Evaluation | transparent v1.9 benchmark + Markdown-aware citation claims + grounded-absence robustness + source/chunk ranking metrics + hard-mode + adaptive-context/compression/scale ablations + planner/web policy + typed Text2SQL + pacing-corrected node latency + calibrated Deep judge | separates retrieval, orchestration, context economics, scale robustness, analysis and generation failures |
| Saved evaluation history | latest Quick/Standard/Deep cache plus timestamped archived runs and deltas | allows instant run switching and lightweight within-workspace regression tracking without consuming Gemini quota again |
| Evaluation report serialization | JSON-safe normalization at save/load/UI/API boundaries + formatted raw JSON view | prevents `root={...}` wrapper leakage and keeps fresh/restored reports identical |
| Evaluation table export | CSV/TSV/Markdown export for every benchmark table | makes Text2SQL/planner/QA/overview/ablation/abstention/comparison results easy to copy or download |
| Incremental Deep evaluation | compatible Standard deterministic baseline + judge-only Deep delta | cuts repeated Standard -> Deep Gemini usage from a full benchmark rerun to the representative judge sample |
| Adaptive reranking | runtime skips cross-encoder for Fast/small/easy cases while retaining explicit ablation and harder-query support | converts measured latency-vs-quality evidence into a runtime optimization without deleting the reranker capability |
| Adaptive focused context pruning | focused semantic/hierarchical fact lookups keep a three-chunk safety floor; broad tasks bypass pruning | improves prompt efficiency while protecting overview, insight and multi-source breadth |
| Context-budget ablation | full top-k vs focused pruning with Precision/Recall/Hit/MRR and context size/token metrics | proves that pruning preserves retrieval quality before treating token reduction as a win |
| Prompt-budget observability | traces expose manifest inclusion, evidence-context chars and estimated generation-prompt tokens | connects latency/cost changes to actual model-input changes |
| Citation repair | conservative lexical evidence matching for uncited factual units, preamble-skip policy and grouped-citation normalization, zero extra LLM calls | improves citation completeness while reducing incorrect or redundant automatically attached citations |
| Evaluation quota control | shared rolling Gemini request ledger, configurable target RPM, provider retry-delay handling and request/pacing telemetry | prevents Standard/Deep benchmark bursts from repeatedly exhausting low free-tier RPM quotas |
| Quality gates | subsystem thresholds cap the letter grade | prevents a strong weighted average from hiding weak Text2SQL/routing/citation behavior |
| Evaluation diagnostics | structured warnings/recommendations for citations, planner taxonomy, Text2SQL and reranker tradeoffs | turns benchmark output into actionable engineering feedback |
| API | FastAPI session/status/ingest/query/evaluate/benchmark-info/saved-evaluations/health/metrics + Swagger/OpenAPI | usable beyond the UI and introspectable from the Architecture + API tab |
| Query/evaluation run state | dedicated status lines + disabled buttons while work is active | prevents silent waits and accidental duplicate submissions without reintroducing overlapping progress overlays |
| Browser session continuity | `gr.BrowserState` stores only the opaque workspace ID | refreshes can reconnect without storing corpus data client-side |
| Lazy demo recovery | empty demo workspace rebuilds on first non-Web question | public demo remains usable after refresh/Space restart without manual lifecycle knowledge |
| Workspace preflight | empty document/table state short-circuits before retrieval/generation | lifecycle failures are not mistaken for retrieval failures |
| Explicit abstain node | LangGraph terminal path for missing local data / insufficient corpus-only evidence | avoids pointless verification/revision and evidence-free generations |
| Inspector summary | workspace stats + semantic plan + evidence + execution path + node-latency waterfall above raw trace | makes architecture and latency bottlenecks legible without hiding low-level diagnostics |
| Source preview hygiene | word-boundary truncation with explicit `...` | prevents visibly broken snippets such as half a final word |
| UI | Gradio demo corpus + RAG switches + progress + trace | recruiters can exercise the architecture without setup |
| Deployment | one Docker Space, non-root writable workspace/cache paths | cheap reproducible hosting without permission failures |
| CI | Ruff + pytest | software-engineering discipline |

## Deliberate production trade-offs

The demo uses embedded Qdrant, in-memory DuckDB, deterministic source profiles, an in-process cache and ephemeral sessions because a small public Hugging Face Space should stay cheap and understandable. The interfaces are intentionally separable so a real service can swap in managed Qdrant, Redis, object storage, a warehouse, OIDC and queued ingestion.

| Profile benchmark | optional Fast/Balanced/Agentic labeled comparison | quantifies whether additional planning, reranking and verification cost is justified for representative tasks |
| Hard-mode robustness | paraphrase/distractor/missing/multi-hop/insight/SQL/local-freshness/injection cases | prevents a near-perfect easy benchmark from becoming meaningless as a regression signal |

| Evaluation provenance | run ID + server-boot ID + fresh/reused status | makes cache/rebuild behavior auditable and prevents fresh runs from being mistaken for saved reuse |
| Context-efficiency diagnostics | source Precision@5 beside Recall@5 + targeted recommendations | exposes distractor-heavy context without sacrificing overview/synthesis breadth prematurely |
| Grounded absence | evidence-cited missing-information answers skip unnecessary revise | rewards calibrated uncertainty and reduces extra model calls on unanswerable local questions |