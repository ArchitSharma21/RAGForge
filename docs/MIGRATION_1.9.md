# Migration to RAGForge v1.9

## Scope

v1.9 is a larger scale/efficiency/readiness release over v1.8. It adds adaptive retrieval depth, dynamic 2-5 chunk context budgets, focused evidence compression, zero-Gemini scale stress, prompt economics, workspace diagnostics, overview/insight consistency and release-readiness evaluation. There are no new runtime dependencies.

## Saved evaluation compatibility

The benchmark version is `1.9`. Existing v1.8 reports remain historical and viewable but are not reused as current v1.9 Standard/Deep baselines because the context/compression/stress methodology and overview semantics changed.

## Runtime behavior

`use_context_pruning` remains the master switch. New `use_adaptive_top_k` and `use_evidence_compression` settings default to enabled. Broad overview/insight/comparison/cross-document tasks preserve evidence breadth. Focused local semantic/hierarchical work may use adaptive 2-5 chunk budgets and generator-only sentence compression.

## Evaluation

Standard/Deep now add local context/compression/scale ablations and release-readiness gates without increasing Gemini request count. Node-latency summaries subtract deliberate quota pacing from service-node time while keeping wall/pacing telemetry separately. Quick intentionally skips these ablation gates.

## API

A new `GET /api/v1/session/{session_id}/diagnostics` endpoint exposes workspace scale/capacity, estimated index memory, TTL/idle state, index readiness and evaluation-history information.

## Hugging Face upgrade

Apply the patch over v1.8, commit the changed files and rebuild the Space. The patch archive does not contain the bundled NIST PDF, so existing LFS/Xet handling is unchanged.
