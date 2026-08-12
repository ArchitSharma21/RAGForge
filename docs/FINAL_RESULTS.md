# Benchmark notes

This document records the main observations from the bundled RAGForge regression suite. The suite uses a five-source demo corpus with one structured table and a 90-chunk base index. It is intended for regression testing and design comparisons, not as a claim about general RAG accuracy.

## Final acceptance note

The v2.0 deployment run exposed a bug in the benchmark itself. A focused QA generation answered the Sev-1 acknowledgement question with `15 minutes`, while the source states `5 minutes`. The old matcher accepted that answer because one accepted phrase was `5 min`, which appeared as a substring of `15 minutes`.

v2.0.1 changed answer matching to respect token/phrase boundaries. v2.0.2 additionally rejects contradictory primary numeric answers even when a later note contains the expected value. v2.0.3 fixes the focused sentence-selection edge case that caused the Sev-2 15-minute target to outrank the Sev-1 5-minute fact for the exact Sev-1 query. Saved v2.0/v2.0.1 results should therefore be treated as historical rather than authoritative v2.0.3 results.

The same run confirmed that the multi-source Acme-vs-OrbitPay timing comparison now follows the intended `Auto + Balanced` comparison path and returns both required time windows correctly.

## Retrieval and context observations

The last full Standard run before the matcher fix showed strong source recall on the demo cases, but lower source precision. That pattern motivated context selection rather than another retrieval layer: retrieve enough evidence to keep the relevant source, then reduce what reaches generation.

The context ablation compared three policies:

| Policy | Median chunks sent to generation | Source Recall@5 | Median estimated context tokens |
|---|---:|---:|---:|
| Full top-k | 6 | 100% | 2,101 |
| Fixed 3-chunk budget | 3 | 100% | 977 |
| Adaptive budget | 4 | 100% | 1,099 |

The adaptive policy remains the runtime default because it keeps more safety margin for ambiguous queries than a global fixed-three rule.

## Sentence compression

Focused sentence selection is applied after the context budget. In the latest ablation, labeled answer evidence was retained in all nine focused cases while the selected evidence was reduced further before generation.

This result is specific to the bundled benchmark. It should be read as evidence that the local compression policy is safe for these cases, not as a universal token-reduction guarantee.

## Synthetic scale stress

The scale harness reuses existing vectors and adds deterministic distractor copies so the real dense + BM25 + RRF retrieval path can be tested at larger index sizes without additional model calls.

| Scale | Chunks | Sources | Source Recall@5 | Hit@1 | MRR |
|---|---:|---:|---:|---:|---:|
| Base | 90 | 5 | 100% | 100% | 1.000 |
| +4x long-document distractors | 434 | 9 | 100% | 100% | 1.000 |
| +19x long-document distractors | 1,724 | 24 | 100% | 87.5% | 0.938 |

The largest synthetic run retained Recall@5, while ranking quality at the very top weakened slightly. This is useful as a regression signal, but a real large-corpus upload is still needed before making stronger scale claims.

## Reranker tradeoff

On the bundled corpus, the cross-encoder reranker repeatedly added multi-second retrieval cost without improving labeled source- or chunk-level ranking metrics. The runtime therefore skips it for the measured small-corpus case and leaves it available for larger or harder workloads.

## Latency

Generation remains the dominant part of service latency. That is why recent optimization work focused on context size and prompt inputs rather than shaving milliseconds from retrieval.

The evaluation UI separates service latency from deliberate request pacing, so a quota-safe Standard run can take several minutes wall-clock without making the pipeline itself appear artificially slow.

## How to use these results

The benchmark is most useful for three things:

1. catching regressions in routing, retrieval, grounding, SQL, and missing-information behavior;
2. comparing runtime policies such as reranking, context budgets, and sentence compression; and
3. exposing measurement bugs in the evaluator itself.

For a final deployment result, rerun Standard after installing v2.0.3 so the contradiction-aware numeric answer matcher is used.
