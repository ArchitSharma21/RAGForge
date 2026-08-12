# Migration to RAGForge v1.8

v1.8 is a context-budget and generation-efficiency release over v1.7. Runtime dependency pins are unchanged.

## What changes

- focused local lookups can conservatively prune the retrieved context to a three-chunk safety floor;
- broad overview, comparison, cross-document and insight/analytical tasks retain full breadth;
- focused fact generation omits the session corpus manifest after routing has already resolved scope;
- traces expose pre/post chunks, sources, estimated tokens, reduction percentage and generation-prompt size;
- Standard/Deep add a zero-Gemini context-budget ablation;
- the Hard Mode missing-answer evaluator accepts the explicit `grounded_absence` trace signal in addition to natural-language absence cues;
- the reranker diagnostic is aligned with the existing small-corpus adaptive skip policy.

## Saved evaluation compatibility

Benchmark version is `1.8`. v1.7 reports remain historical and can still be viewed, but are not eligible for automatic reuse under v1.8 because the evaluation report now includes context-budget metrics and the Hard Mode missing-answer decision semantics are hardened.

## Suggested validation

1. Ask a simple focused document question and verify the inspector shows a context-budget decision such as `6 -> 3` chunks.
2. Ask an overview/insight question and confirm pruning is skipped with a breadth-preserving reason.
3. Run Standard once and inspect the new **Context budget** table. Recall@5 should remain unchanged before the pruning policy is considered successful.
4. Confirm `hard_missing_orbitpay_fee` reports either `missing_answer_match=true` or `grounded_absence=true` and passes.
