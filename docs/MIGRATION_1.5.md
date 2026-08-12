# Migration to v1.5

v1.5 is an evidence-driven optimization/evaluation release over v1.4.1. It does not change the bundled demo corpus or dependency pins.

## Main changes

- per-workspace saved Quick/Standard/Deep evaluation history;
- instant saved-run switching and side-by-side comparison;
- optional zero-call reuse of an already compatible evaluation;
- incremental Deep evaluation that reuses a compatible Standard deterministic baseline;
- typed scalar Text2SQL benchmark checks;
- adaptive cross-encoder reranking based on profile/task/corpus complexity;
- deterministic evidence-aware citation repair with no extra model call;
- table planner examples distinguishing direct lookups from cross-row aggregation;
- saved-evaluation REST endpoints and updated Architecture + API tab.

## Deployment

Apply the v1.5 patch over a clean v1.4.1 project and commit normally. The patch does not contain the bundled NIST PDF, so the existing Hugging Face Xet/LFS setup is unchanged.

After deployment:

1. index the demo corpus;
2. run Quick, then Standard;
3. leave `Reuse saved evaluation` enabled and run Deep;
4. confirm Deep reports `reused_standard_baseline: true` and only the sampled judge requests;
5. switch `View saved evaluation` among Quick/Standard/Deep and inspect `Compare saved runs`;
6. verify the Text2SQL boolean case reports `match_method=typed_scalar`, `observed_value=true`, `expected_value=true`;
7. ask a normal focused document question and inspect `reranker_used`/`reranker_reason` in the trace.
