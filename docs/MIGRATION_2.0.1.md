# Upgrade to RAGForge v2.0.1

v2.0.1 is a small correctness and presentation patch over v2.0.0.

## What changed

- landing and evaluation UI copy is simplified and historical benchmark badges are removed
- evaluation summaries are scoped to the bundled benchmark instead of presenting grade/readiness cards
- Quick no longer reports acceptance/readiness results for ablations it does not run
- answer-key matching is boundary-aware, fixing numeric substring false positives such as `5 min` matching `15 minutes`
- the source-localization planner case accepts both semantic and hierarchical retrieval
- context-budget experiment names no longer contain development-version labels
- README and evaluation documentation are rewritten around behavior, methodology, and limitations

## Benchmark compatibility

The benchmark version is `2.0.1`. Existing `2.0` saved evaluation reports remain historical but are not reused as current results because answer-matching semantics changed.

## Upgrade

Apply the patch over v2.0.0, then run Quick. Run Standard once if you want a new authoritative benchmark result under the corrected matcher.
