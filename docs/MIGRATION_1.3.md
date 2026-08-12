# Migration: v1.2 -> v1.3

v1.3 is a normal source/UI/evaluation upgrade. No database or corpus migration is required.

## Main changes

- hides Gradio's automatic full event overlay on long-running handlers and keeps one explicit `gr.Progress` surface,
- reports query stages directly from LangGraph nodes,
- adds trace efficiency metrics and estimated LLM-call counts,
- replaces the minimal Evaluation JSON view with a scorecard and layer-specific tables,
- adds Quick / Standard / Deep evaluation modes,
- adds `evals/demo_benchmark.json` and retrieval reranker ablations,
- adds optional Gemini faithfulness/relevance/completeness/citation-support judging,
- retains all v1.2 browser-session, lazy-demo, preflight and abstention behavior.

## Upgrade

Extract the v1.3 patch over the v1.2 repository, then:

```bash
git add .
git commit -m "Upgrade RAGForge to v1.3 evaluation and UI polish"
git push origin main
```

The patch does not need to replace the bundled NIST PDF, so an existing Xet/LFS setup can remain unchanged.

## Post-deploy checks

1. Click **Index corpus** and confirm only one progress UI is visible.
2. Ask a normal corpus question and confirm retrieval/generation stages appear once.
3. Run **Quick** evaluation and confirm the scorecard/tables populate.
4. Run **Standard** evaluation and inspect the retrieval ablation.
5. Optionally run **Deep** to verify Gemini judge fields appear separately from the deterministic score.
