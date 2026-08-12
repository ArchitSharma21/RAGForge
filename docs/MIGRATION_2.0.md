# Migration to RAGForge v2.0.0

v2.0 is the final planned feature release over v1.9. It is primarily a productization and benchmark-alignment release.

## Runtime behavior

The retrieval/generation architecture from v1.9 is preserved:

- corpus-scale adaptive retrieval depth
- dynamic 2-5 chunk focused context budgets
- focused evidence sentence compression
- source-balanced/global/hierarchical/analytical retrieval
- adaptive reranker policy
- CRAG/Self-RAG controls
- Text2SQL and conditional web research

The Hard Mode multi-hop comparison case is now evaluated through `Auto + Balanced` and must resolve to `comparison -> hierarchical`. This aligns the benchmark with the recommended semantic path.

## UI

- final v2.0 product hero and measured-baseline cards
- Chat/Evaluation hierarchy simplified
- recommended profile guidance surfaced
- evaluation score-card metric tiles
- duplicate Architecture runtime header fixed
- final project-status footer

## Documentation/release tooling

- README rewritten as a finished product page
- `CHANGELOG.md` added
- `docs/FINAL_RESULTS.md` added
- `docs/PORTFOLIO_GUIDE.md` added
- `scripts/release_check.py` added
- `make smoke` and `make verify` added
- CI now runs compilation and the release consistency checker

## Benchmark compatibility

The benchmark version is `2.0`. Existing v1.9 saved reports remain historical and are not reused as current v2.0 Standard/Deep baselines.

No new runtime dependency is introduced.
