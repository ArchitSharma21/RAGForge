# Changelog

## 2.0.3

- Fixed focused evidence compression for sibling entity facts such as Sev-1 vs Sev-2 targets.
- Generic follow-up sentences now inherit nearby entity context during deterministic sentence ranking.
- Added a regression test for the exact Sev-1/15-minute failure observed in the deployed Quick benchmark.


## v2.0.1

- simplified the product UI and removed benchmark/grade marketing from the landing and evaluation screens
- rewrote the README and evaluation notes around system behavior, methodology, and limitations
- fixed numeric answer-key matching so labels such as `5 min` cannot match `15 minutes`
- changed Quick evaluation so skipped scale/context checks are reported as not run rather than as release readiness
- allowed semantic or hierarchical retrieval for the source-localization planner case, since both are valid strategies
- renamed version-specific context-ablation labels to neutral experimental names

## v2.0.0

- consolidated the UI, documentation, architecture diagram, and release tooling
- aligned the multi-source Hard Mode comparison with the recommended Auto + Balanced path
- added release consistency checks and CI verification

## v1.9

- added scale-aware retrieval depth and adaptive focused context budgets
- added focused sentence selection and synthetic distractor scale stress
- added prompt/context telemetry and workspace diagnostics

## v1.8

- added focused context pruning and context-budget ablation

## v1.7

- improved evaluator correctness, citation handling, source rendering, and runtime provenance
- adopted the measured small-corpus reranker skip policy

## v1.6

- added analytical corpus synthesis, structured table evidence, harder robustness cases, and chunk-level reranker evaluation

## v1.5

- added reusable Quick/Standard/Deep reports, incremental Deep evaluation, typed Text2SQL checks, and adaptive reranking

## v1.4

- expanded evaluation metrics, cache-bypassed timing, Architecture/API inspection, and quota-aware evaluation

## v1.3

- introduced component-level evaluation and reranker ablation

## v1.2

- hardened session lifecycle, lazy corpus recovery, progress UI, and explicit abstention

## v1.1

- introduced semantic query planning, source profiles, global/hierarchical retrieval, and correction-before-web policy

## v1.0

- initial FastAPI + Gradio + LangGraph application with hybrid retrieval, Text2SQL, web search, citations, and Docker deployment
