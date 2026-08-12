# Security model

## v2.0 final security posture

v2.0 does not add a new external execution surface. The final release keeps the existing archive, SQL, SSRF, session-isolation, prompt-injection, secret-handling and rate-limit controls, while clarifying operational limits in the README and portfolio documentation. `make verify` now includes a dependency-free release-consistency check so version/benchmark/documentation drift is caught in CI.

The project remains a hardened **portfolio/demo** application, not a compliance-certified multi-tenant SaaS.

## v1.9 resource and scale controls

v1.9 adds explicit workspace-health telemetry for chunk-cap utilization, corpus scale, approximate in-memory vector size, session TTL/idle age and index readiness. Adaptive retrieval depth remains bounded (up to 12 candidates by policy), focused generation budgets remain bounded (2-5 chunks), and sentence compression only changes the generator copy of retrieved text. The synthetic scale-stress evaluator reuses already-computed vectors and is isolated so a local stress-harness failure cannot crash the primary Standard/Deep report.

These controls improve demo predictability but are not substitutes for tenant quotas, durable storage, external vector databases, worker isolation or resource limits in a real multi-user production deployment.

RAGForge is a hardened **portfolio/demo** application, not a compliance-certified multi-tenant SaaS. The goal is to demonstrate the controls a production RAG should think about while keeping the Hugging Face Space inexpensive and understandable.

## Threats covered

| Threat | Control in this repo |
|---|---|
| ZIP-slip / path traversal | archive members with absolute paths or `..` are rejected; filenames are sanitized |
| Archive bombs | compressed upload, file-count, and total uncompressed-size limits |
| Arbitrary file ingestion | extension allow-list; nested archives are not extracted |
| Prompt injection in retrieved data | document/web content is explicitly treated as untrusted; suspicious chunks are scored/down-weighted; suspicious chunks are excluded from source-profile/manifest excerpts when possible; planner and answer prompts never treat retrieved instructions as system instructions |
| SQL mutation / exfiltration | isolated in-memory DuckDB; only one `SELECT`/CTE is accepted; mutation/admin keywords are rejected; a result limit is enforced |
| SSRF from web results | only HTTP(S), public-resolving hosts are accepted; localhost/private/link-local/reserved/multicast addresses are rejected; redirects are disabled during page fetch |
| Cross-user retrieval leakage | per-session workspace, vector index, DuckDB database, history, cache namespace, and session TTL |
| Browser session identifier | only an opaque workspace ID is stored in `gr.BrowserState`; document text, embeddings, API keys and SQL data remain server-side |
| Concurrent index corruption | per-workspace re-entrant lock serializes ingestion/query mutations; shared cache has its own lock |
| API quota abuse | per-IP sliding-window query limiter in UI/REST; optional REST Bearer token; evaluation uses a separate rolling per-key/per-model Gemini request budget with configurable RPM pacing and bounded 429 backoff |
| Secret leakage | `.env` ignored; secrets are expected through Hugging Face Space Secrets or user-entered key |
| Unbounded context | chunk/session limits, top-k limits, compact/truncated corpus manifest, source truncation before generation |
| Accidental external-data leakage | semantic planner separates corpus/external/mixed scope; web fallback requires both permission and semantic relevance; corpus-only retrieval failure can abstain rather than automatically search the web |

## Saved evaluation data

v1.5 stores the latest Quick, Standard and Deep evaluation reports inside the current server-side workspace so users can compare runs and Deep can reuse the exact Standard deterministic baseline. Reports may contain generated answers, retrieved source snippets and benchmark metadata. They follow the same session TTL, reset behavior and ephemeral container lifecycle as the corpus, are not written to browser storage, and are not intended as durable audit storage.

## Important residual risks

- The SSRF filter resolves a hostname before fetching it, but a sophisticated DNS-rebinding setup can still be a risk in generic URL-fetching systems. For an enterprise deployment, use an outbound proxy/egress allow-list and network policy rather than application checks alone.
- Extension checks are not content-type malware scanning. Do not accept untrusted executable formats in a real document-processing service; add MIME sniffing, AV scanning, sandboxed parsers, and object-storage quarantine.
- Browser persistence does not make the corpus durable. A container restart invalidates the server-side workspace even if the browser still has the old opaque ID; demo mode can rebuild bundled files, but custom uploads must be re-indexed.
- The demo uses in-process sessions and one application process. Multi-replica deployments need durable tenant/session state and tenant filters enforced at the storage layer.
- Basic prompt-injection detection is heuristic. Source-profile excerpts are derived from untrusted files and therefore remain an indirect-injection surface even with filtering/system instructions. Treat these controls as defense-in-depth, not a proof of safety.
- Free Gemini API tiers may have different data-use terms from paid tiers. Do not put confidential documents into a public demo or a provider tier whose privacy terms do not meet your requirements. Active provider quotas can also change or vary by project/model; quota-safe pacing reduces burst failures but cannot create quota that the provider has not granted.

## Reporting

If you publish a fork, replace this section with your preferred vulnerability-reporting contact and do not ask reporters to open public issues for secrets or exploitable vulnerabilities.


## Evaluation exports

CSV/TSV/Markdown evaluation exports are generated only from the current workspace's benchmark
results and are written under that workspace's ephemeral evaluation directory. Filenames are
sanitized from a fixed UI table label and evaluation depth; user-supplied paths are not used.

## v1.6 analytical-synthesis security notes

- Analytical synthesis never executes arbitrary model-authored SQL. The `[T#]` evidence path uses deterministic read-only DuckDB inspection (schema, bounded rows and descriptive signals) before the grounded generation step.
- The existing Text2SQL path still validates model-authored SQL as a single read-only `SELECT`/CTE before execution.
- Semantic citation attribution uses the already-loaded local embedding model only as a high-threshold fallback. It can attach a citation label, but it cannot change source text, execute instructions, or grant retrieved content instruction priority.
- Hard Mode includes an explicit prompt-injection detector case so regressions in stored-instruction detection remain visible in the evaluation report.

## Evaluation provenance (v1.7)

Saved evaluation metadata contains only operational provenance (short run ID, server-boot ID, model, benchmark version, corpus version and timestamp). It does not add user document content to browser storage. The opaque browser session ID remains the only client-persisted workspace identifier.

## Context pruning safety (v1.8)

Focused context pruning is intentionally post-retrieval and task-scoped. It does not weaken prompt-injection scanning, source access controls, archive hardening or web SSRF protections. Broad/multi-source/analytical tasks bypass pruning, and the three-chunk safety floor is evaluated against source recall before the optimization is treated as successful.