from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .context_budget import adaptive_context_budget, adaptive_retrieval_top_k
from .eval_metrics import mean, source_metrics
from .retrieval import HybridRetriever
from .schemas import Chunk, PipelineConfig, QueryPlan
from .workspace import Workspace


@dataclass(slots=True)
class StressCorpus:
    chunks: list[Chunk]
    vectors: np.ndarray
    distractor_copies: int


def _clone_distractors(workspace: Workspace, copies: int) -> StressCorpus | None:
    base_chunks = list(workspace.retriever.chunks)
    base_vectors = workspace.retriever._vectors  # internal by design for zero-reembedding stress evaluation
    if not base_chunks or base_vectors is None or len(base_chunks) != len(base_vectors):
        return None

    distractor_indices = [
        idx for idx, chunk in enumerate(base_chunks)
        if "NIST_AI_RMF" in chunk.source
    ]
    if not distractor_indices:
        # Fall back to the longest source so the stress corpus still simulates a
        # large repeated distractor document for non-matching QA cases.
        counts: dict[str, int] = {}
        for chunk in base_chunks:
            counts[chunk.source] = counts.get(chunk.source, 0) + 1
        if not counts:
            return None
        source = max(counts, key=counts.get)
        distractor_indices = [i for i, chunk in enumerate(base_chunks) if chunk.source == source]

    chunks = list(base_chunks)
    vectors = [np.asarray(row, dtype=np.float32) for row in base_vectors]
    for copy_idx in range(max(0, int(copies))):
        for idx in distractor_indices:
            original = base_chunks[idx]
            chunks.append(
                Chunk(
                    id=f"stress-{copy_idx:03d}-{original.id}",
                    text=original.text,
                    source=f"stress_distractor_{copy_idx:03d}.pdf",
                    page=original.page,
                    section=original.section,
                    metadata={**original.metadata, "synthetic_stress_distractor": True},
                )
            )
            vectors.append(np.asarray(base_vectors[idx], dtype=np.float32))
    return StressCorpus(chunks=chunks, vectors=np.vstack(vectors), distractor_copies=max(0, int(copies)))


def scale_stress_retrieval_eval(
    workspace: Workspace,
    qa_cases: list[dict[str, Any]],
    *,
    levels: tuple[int, ...] = (0, 4, 19),
) -> list[dict[str, Any]]:
    """Stress hybrid retrieval with a 1x/5x/20x long-document distractor corpus.

    The added chunks are cloned from the long NIST source but renamed as
    synthetic distractor sources. NIST-labeled QA cases are excluded so the
    clones cannot accidentally count as relevant. Existing vectors are reused,
    making the stress test deterministic and zero-Gemini.
    """
    eligible = [
        case for case in qa_cases
        if not any("NIST_AI_RMF" in str(source) for source in case.get("relevant_sources", []))
    ]
    if not eligible:
        return []

    rows: list[dict[str, Any]] = []
    for copies in levels:
        stress = _clone_distractors(workspace, copies)
        if stress is None:
            return []
        retriever = HybridRetriever(collection=f"stress_{copies}")
        build_started = time.perf_counter()
        retriever.index_precomputed(stress.chunks, stress.vectors)
        build_ms = (time.perf_counter() - build_started) * 1000
        source_count = len({chunk.source for chunk in stress.chunks})

        metric_rows: list[dict[str, float]] = []
        latencies: list[float] = []
        budget_targets: list[float] = []
        context_tokens: list[float] = []
        pruning_recall: list[float] = []
        for case in eligible:
            plan = QueryPlan(
                route="documents",
                knowledge_scope="corpus",
                task_type="fact_lookup",
                retrieval_strategy="semantic",
                web_relevance="irrelevant",
                rewritten_query=case["question"],
                document_queries=[case["question"]],
            )
            cfg = PipelineConfig(
                profile="Balanced",
                top_k=6,
                use_reranker=False,
                use_context_pruning=True,
                use_adaptive_top_k=True,
            )
            effective_k = adaptive_retrieval_top_k(
                cfg,
                plan,
                corpus_chunks=len(stress.chunks),
                corpus_sources=source_count,
            )
            started = time.perf_counter()
            hits = retriever.search(case["question"], top_k=effective_k, use_reranker=False)
            latencies.append((time.perf_counter() - started) * 1000)
            raw_sources = [hit.chunk.source for hit in hits[:5]]
            metric_rows.append({k: float(v) for k, v in source_metrics(raw_sources, case.get("relevant_sources", [])).items()})

            budget = adaptive_context_budget(
                hits,
                plan,
                cfg,
                corpus_chunks=len(stress.chunks),
                corpus_sources=source_count,
            )
            budget_targets.append(float(budget.target_chunks))
            context_tokens.append(float(budget.tokens_est_after))
            pruned_sources = [hit.chunk.source for hit in budget.hits[:5]]
            pruning_recall.append(float(source_metrics(pruned_sources, case.get("relevant_sources", []))["source_recall@5"]))

        rows.append(
            {
                "scale_label": "1x base" if copies == 0 else f"+{copies}x long-doc distractors",
                "distractor_copies": copies,
                "chunks": len(stress.chunks),
                "sources": source_count,
                "source_precision@5": round(mean([r["source_precision@5"] for r in metric_rows]), 3),
                "source_recall@5": round(mean([r["source_recall@5"] for r in metric_rows]), 3),
                "source_hit@1": round(mean([r["source_hit@1"] for r in metric_rows]), 3),
                "source_mrr": round(mean([r["source_mrr"] for r in metric_rows]), 3),
                "adaptive_pruned_recall@5": round(mean(pruning_recall), 3),
                "median_adaptive_budget_chunks": round(statistics.median(budget_targets), 1) if budget_targets else 0.0,
                "median_context_tokens_est": round(statistics.median(context_tokens), 1) if context_tokens else 0.0,
                "median_retrieval_ms": round(statistics.median(latencies), 1) if latencies else 0.0,
                "index_build_ms": round(build_ms, 1),
                "cases": len(eligible),
            }
        )
    return rows
