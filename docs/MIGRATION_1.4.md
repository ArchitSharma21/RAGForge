# Migration: v1.3 -> v1.4

v1.4 is a source/UI/evaluation upgrade. No corpus or database migration is required.

## Main changes

- Ask immediately shows `Processing...` and disables duplicate submissions.
- Evaluation immediately shows a running state and disables duplicate benchmark runs.
- Source-level AP/nDCG metrics deduplicate repeated chunks from one file.
- Evaluation bypasses response caching and does not mutate chat history.
- Deep judge scores are citation-calibrated.
- Letter grades use quality gates.
- Structured table lookup routing is clarified.
- The demo benchmark expands to v1.4.
- Architecture + API becomes an interactive runtime/API reference tab.
- New session-status and benchmark-metadata API endpoints are available.

## Upgrade

Extract the v1.4 patch over a clean v1.3 repository, then:

```bash
git add .
git commit -m "Upgrade RAGForge to v1.4 evaluation and API observability"
git push origin main
```

The patch intentionally excludes `demo_documents/NIST_AI_RMF_1.0.pdf`, so an existing Hugging Face Xet/LFS setup is not disturbed.

## Post-deploy checks

1. Click Ask and confirm the button changes to `Processing...` immediately.
2. Start Quick evaluation and confirm `Run evaluation` becomes disabled with a visible running message.
3. Run Standard and verify source AP@5 never exceeds 1.0.
4. Run Standard followed by Deep and verify Deep pipeline latency is not near zero due to cache hits.
5. Inspect diagnostics for Text2SQL/citation/reranker findings.
6. Open Architecture + API, click `Refresh runtime view`, and verify the current workspace and curl examples appear.
7. Open `/docs` to confirm the FastAPI v1.4 endpoints.
