# Migration to RAGForge v1.7

v1.7 is a correctness, provenance and measured-efficiency release over v1.6. Runtime dependency pins are unchanged.

## Highlights

- escaped uniform source cards;
- graphical node-latency waterfall;
- Markdown-aware citation coverage;
- grounded missing-information answers and no unnecessary revise call;
- valid `[T#]` evidence on corpus overviews;
- evaluation run IDs/server-boot provenance and user-input-only saved-run switching;
- profile-policy and context-efficiency diagnostics;
- small-corpus reranker skip extended to Agentic after source+chunk ablation showed no gain;
- benchmark version `1.7`.

Because citation/missing-answer scoring semantics changed, v1.6 saved reports remain historical and are not eligible for v1.7 automatic benchmark reuse.

Apply the patch, rebuild the Space, index the demo corpus, run Quick first, then Standard. A fresh run should say `Fresh ... evaluation complete`, while an explicit saved-run load should say `Loaded saved ... run`.
