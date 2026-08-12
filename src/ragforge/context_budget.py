from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .schemas import PipelineConfig, QueryPlan, SearchHit


@dataclass(slots=True)
class ContextBudgetDecision:
    hits: list[SearchHit]
    used: bool
    reason: str
    policy: str
    target_chunks: int
    corpus_scale: str
    retrieval_confidence: float
    score_gap: float
    chunks_before: int
    chunks_after: int
    sources_before: int
    sources_after: int
    chars_before: int
    chars_after: int
    tokens_est_before: int
    tokens_est_after: int

    @property
    def reduction_ratio(self) -> float:
        if self.chars_before <= 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - (self.chars_after / self.chars_before)))

    def trace_fields(self) -> dict[str, object]:
        return {
            "context_pruning_used": self.used,
            "context_pruning_reason": self.reason,
            "context_budget_policy": self.policy,
            "context_budget_target_chunks": self.target_chunks,
            "corpus_scale": self.corpus_scale,
            "retrieval_confidence": round(self.retrieval_confidence, 3),
            "retrieval_score_gap": round(self.score_gap, 3),
            "context_chunks_before": self.chunks_before,
            "context_chunks_after": self.chunks_after,
            "context_sources_before": self.sources_before,
            "context_sources_after": self.sources_after,
            "context_chars_before": self.chars_before,
            "context_chars_after": self.chars_after,
            "context_tokens_est_before": self.tokens_est_before,
            "context_tokens_est_after": self.tokens_est_after,
            "context_reduction_pct": round(self.reduction_ratio * 100.0, 1),
        }


def estimate_tokens_from_chars(chars: int) -> int:
    """Cheap deterministic token estimate for observability, not billing."""
    return int(math.ceil(max(0, int(chars)) / 4.0))


def corpus_scale_label(chunks: int, sources: int) -> str:
    chunks = max(0, int(chunks))
    sources = max(0, int(sources))
    if chunks >= 3000 or sources >= 80:
        return "very_large"
    if chunks >= 1000 or sources >= 30:
        return "large"
    if chunks >= 250 or sources >= 10:
        return "medium"
    return "small"


def adaptive_retrieval_top_k(
    config: PipelineConfig,
    plan: QueryPlan,
    *,
    corpus_chunks: int,
    corpus_sources: int,
) -> int:
    """Choose initial retrieval breadth from corpus scale and task breadth.

    ``top_k`` remains the small-corpus baseline configured by the user. Larger
    corpora can retrieve a wider candidate set before the focused context
    budget trims the generation context. Broad tasks retain wider evidence.
    """
    baseline = max(2, min(12, int(config.top_k)))
    if not getattr(config, "use_adaptive_top_k", True):
        return baseline

    scale = corpus_scale_label(corpus_chunks, corpus_sources)
    scale_floor = {"small": baseline, "medium": max(baseline, 8), "large": max(baseline, 10), "very_large": 12}[scale]

    if plan.task_type in {"overview", "insight_synthesis", "cross_document_synthesis", "comparison"}:
        # Broad tasks need source breadth. Keep the absolute cap bounded by the
        # schema and avoid forcing more chunks than sources can usefully supply.
        broad_floor = min(12, max(scale_floor, min(max(6, corpus_sources), 12)))
        return broad_floor
    return min(12, scale_floor)


def _context_chars(hits: Iterable[SearchHit]) -> int:
    # Include a small deterministic metadata allowance per chunk because the
    # generation context contains source/page labels in addition to raw text.
    return sum(len(hit.chunk.text or "") + len(hit.chunk.source or "") + 24 for hit in hits)


def _source_count(hits: Iterable[SearchHit]) -> int:
    return len({hit.chunk.source for hit in hits})


def _hit_signal(hit: SearchHit) -> float:
    dense = max(0.0, min(1.0, float(hit.dense_score or 0.0)))
    sparse = max(0.0, min(1.0, float(hit.sparse_score or 0.0))) * 0.9
    fused = max(0.0, min(1.0, float(hit.score or 0.0)))
    return max(dense, sparse, fused)


def _retrieval_confidence(hits: list[SearchHit]) -> tuple[float, float]:
    if not hits:
        return 0.0, 0.0
    signals = [_hit_signal(hit) for hit in hits[:4]]
    top = signals[0]
    second = signals[1] if len(signals) > 1 else 0.0
    gap = max(0.0, top - second)
    # Agreement between dense and sparse on the first result adds confidence,
    # while a small top-two gap indicates ambiguity even with a strong top hit.
    first = hits[0]
    method_agreement = float(
        float(first.dense_score or 0.0) >= 0.20 and float(first.sparse_score or 0.0) >= 0.05
    )
    confidence = 0.65 * top + 0.25 * min(1.0, gap / 0.20) + 0.10 * method_agreement
    return max(0.0, min(1.0, confidence)), gap


def adaptive_context_budget(
    hits: list[SearchHit],
    plan: QueryPlan,
    config: PipelineConfig,
    *,
    corpus_chunks: int = 0,
    corpus_sources: int = 0,
) -> ContextBudgetDecision:
    """Adaptively shrink context for focused local lookups only.

    v1.8 proved that a three-chunk safety floor could cut the demo context by
    roughly half without harming source recall. v1.9 generalizes that policy:
    the target is chosen from retrieval confidence, score separation and corpus
    scale. Broad/synthesis tasks are never pruned by this function.
    """
    before = list(hits or [])
    chunks_before = len(before)
    chars_before = _context_chars(before)
    sources_before = _source_count(before)
    scale = corpus_scale_label(corpus_chunks or chunks_before, corpus_sources or sources_before)
    retrieval_confidence, score_gap = _retrieval_confidence(before)

    def decision(
        after: list[SearchHit],
        used: bool,
        reason: str,
        *,
        target: int,
        policy: str = "adaptive_focused_budget",
    ) -> ContextBudgetDecision:
        chars_after = _context_chars(after)
        return ContextBudgetDecision(
            hits=after,
            used=used,
            reason=reason,
            policy=policy,
            target_chunks=max(0, int(target)),
            corpus_scale=scale,
            retrieval_confidence=retrieval_confidence,
            score_gap=score_gap,
            chunks_before=chunks_before,
            chunks_after=len(after),
            sources_before=sources_before,
            sources_after=_source_count(after),
            chars_before=chars_before,
            chars_after=chars_after,
            tokens_est_before=estimate_tokens_from_chars(chars_before),
            tokens_est_after=estimate_tokens_from_chars(chars_after),
        )

    if not before:
        return decision(before, False, "no_document_evidence", target=0)
    if not getattr(config, "use_context_pruning", True):
        return decision(before, False, "disabled_by_user", target=chunks_before, policy="disabled")
    if plan.route not in {"documents", "hybrid"}:
        return decision(before, False, "non_local_route", target=chunks_before)
    if plan.task_type not in {"fact_lookup", "followup"}:
        return decision(before, False, "broad_or_multi_source_task", target=chunks_before)
    if plan.retrieval_strategy not in {"semantic", "hierarchical"}:
        return decision(before, False, "strategy_requires_breadth", target=chunks_before)

    # Base safety grows with corpus scale because a larger corpus increases the
    # chance that a seemingly focused question needs a runner-up source.
    base = {"small": 3, "medium": 3, "large": 4, "very_large": 5}[scale]
    target = base

    # A clearly separated high-confidence top result can safely use two chunks
    # on small/medium corpora. Avoid top-1 pruning so a second supporting chunk
    # remains available for citation and accidental under-classification.
    if scale in {"small", "medium"} and retrieval_confidence >= 0.78 and score_gap >= 0.15:
        target = 2
    # Ambiguous retrieval needs extra evidence, especially when the top results
    # already span multiple sources.
    top_sources = len({hit.chunk.source for hit in before[:3]})
    if retrieval_confidence < 0.55 or score_gap < 0.035 or top_sources >= 3:
        target = min(5, max(target, base + 1))
    if plan.task_type == "followup":
        target = min(5, max(target, 3))

    # The initial adaptive retrieval depth may exceed config.top_k on a large
    # corpus, so cap against the actual candidate count rather than config.top_k.
    target = max(2, min(target, chunks_before))
    if chunks_before <= target:
        return decision(before, False, "already_within_adaptive_budget", target=target)

    after = before[:target]
    reason = f"adaptive_{scale}_top{target}"
    return decision(after, True, reason, target=target)


def focused_context_budget(
    hits: list[SearchHit],
    plan: QueryPlan,
    config: PipelineConfig,
    *,
    safety_floor: int = 3,
) -> ContextBudgetDecision:
    """Backward-compatible v1.8 fixed safety-floor policy.

    The runtime uses :func:`adaptive_context_budget` in v1.9. Keeping this
    helper preserves the documented v1.8 ablation semantics and makes release
    deltas auditable instead of silently changing an old experiment.
    """
    before = list(hits or [])
    chunks_before = len(before)
    chars_before = _context_chars(before)
    sources_before = _source_count(before)
    confidence, gap = _retrieval_confidence(before)

    def decision(after: list[SearchHit], used: bool, reason: str) -> ContextBudgetDecision:
        chars_after = _context_chars(after)
        return ContextBudgetDecision(
            hits=after,
            used=used,
            reason=reason,
            policy="v1.8_fixed_safety_floor",
            target_chunks=len(after),
            corpus_scale=corpus_scale_label(chunks_before, sources_before),
            retrieval_confidence=confidence,
            score_gap=gap,
            chunks_before=chunks_before,
            chunks_after=len(after),
            sources_before=sources_before,
            sources_after=_source_count(after),
            chars_before=chars_before,
            chars_after=chars_after,
            tokens_est_before=estimate_tokens_from_chars(chars_before),
            tokens_est_after=estimate_tokens_from_chars(chars_after),
        )

    if not before:
        return decision(before, False, "no_document_evidence")
    if not getattr(config, "use_context_pruning", True):
        return decision(before, False, "disabled_by_user")
    if plan.route not in {"documents", "hybrid"}:
        return decision(before, False, "non_local_route")
    if plan.task_type not in {"fact_lookup", "followup"}:
        return decision(before, False, "broad_or_multi_source_task")
    if plan.retrieval_strategy not in {"semantic", "hierarchical"}:
        return decision(before, False, "strategy_requires_breadth")
    target = max(2, min(int(safety_floor), int(config.top_k)))
    if chunks_before <= target:
        return decision(before, False, "already_within_budget")
    return decision(before[:target], True, "focused_lookup_top3_safety_floor")
