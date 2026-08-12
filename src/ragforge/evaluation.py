from __future__ import annotations

import json
import statistics
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from .context_budget import adaptive_context_budget, adaptive_retrieval_top_k, focused_context_budget
from .evidence_compression import focused_evidence_compression
from .eval_metrics import (
    answer_key_match,
    citation_metrics,
    mean,
    missing_answer_match,
    percentile,
    safe_div,
    scalar_value_match,
    source_metrics,
)
from .llm import GeminiGateway, RequestPacer
from .pipeline import RAGEngine
from .schemas import PipelineConfig, QueryPlan
from .security import prompt_injection_score
from .stress_eval import scale_stress_retrieval_eval
from .workspace import Workspace

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = ROOT / "evals" / "demo_benchmark.json"


def _load_benchmark() -> dict[str, Any]:
    return json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))


def demo_benchmark_metadata() -> dict[str, Any]:
    benchmark = _load_benchmark()
    return {
        "version": benchmark.get("version"),
        "description": benchmark.get("description", ""),
        "cases": {
            "focused_qa": len(benchmark.get("qa_cases", [])),
            "semantic_planner": len(benchmark.get("planner_cases", [])),
            "corpus_overview": len(benchmark.get("overview_cases", [])),
            "text2sql": len(benchmark.get("sql_cases", [])),
            "hard_mode": len(benchmark.get("hard_mode_cases", [])),
            "lifecycle_abstention": 2,
        },
        "levels": {
            "Quick": "Small deployment smoke test",
            "Standard": "Full deterministic benchmark, hard-mode robustness, retrieval/context/compression ablations, synthetic scale stress and release-readiness checks",
            "Deep": "Calibrated Gemini judge layered onto Standard; reuses a compatible saved Standard baseline when available",
        },
        "default_target_rpm": 12,
        "deep_judge_cases": sum(
            1 for case in benchmark.get("qa_cases", []) + benchmark.get("overview_cases", []) if case.get("deep_judge")
        ),
        "zero_gemini_ablations": ["reranker", "adaptive-context-budget", "evidence-compression", "scale-stress"],
        "cache_policy": (
            "RAG response cache is bypassed during fresh benchmark execution. Completed Quick/Standard/Deep reports "
            "can be saved per workspace, and Deep can reuse a compatible Standard deterministic baseline."
        ),
    }


def _document_sources(result_sources: list[dict[str, Any]], k: int = 5) -> list[str]:
    return [
        str(source.get("title", ""))
        for source in result_sources
        if source.get("type") == "document"
    ][:k]


def _evidence_text(result_sources: list[dict[str, Any]]) -> str:
    blocks = []
    for source in result_sources:
        sid = source.get("id", "?")
        title = source.get("title", "Source")
        snippet = source.get("snippet", "")
        url = source.get("url")
        blocks.append(f"[{sid}] {title}\n{('URL: ' + url + chr(10)) if url else ''}{snippet}")
    return "\n\n".join(blocks)


def _trace_efficiency(trace: dict[str, Any]) -> dict[str, Any]:
    metrics = trace.get("metrics", {})
    nodes = trace.get("nodes", [])
    names = [str(node.get("node", "")) for node in nodes]
    retrieve = next((node for node in reversed(nodes) if node.get("node") == "retrieve"), {})
    generate = next((node for node in reversed(nodes) if node.get("node") == "generate"), {})
    return {
        "node_count": int(metrics.get("node_count", len(nodes)) or 0),
        "llm_calls_estimate": int(metrics.get("llm_calls_estimate", 0) or 0),
        "web_used": bool(metrics.get("web_used", "web" in names)),
        "correction_used": bool(metrics.get("correction_used", "correct" in names)),
        "abstained": bool(metrics.get("abstained", "abstain" in names)),
        "cache_hit": bool(trace.get("cache_hit", False)),
        "context_pruning_used": bool(retrieve.get("context_pruning_used", False)),
        "context_chunks_before": int(retrieve.get("context_chunks_before", 0) or 0),
        "context_chunks_after": int(retrieve.get("context_chunks_after", 0) or 0),
        "context_tokens_est_before": int(retrieve.get("context_tokens_est_before", 0) or 0),
        "context_tokens_est_after": int(retrieve.get("context_tokens_est_after", 0) or 0),
        "context_reduction_pct": float(retrieve.get("context_reduction_pct", 0.0) or 0.0),
        "manifest_included": bool(generate.get("manifest_included", False)),
        "generation_prompt_tokens_est": int(generate.get("generation_prompt_tokens_est", 0) or 0),
        "generation_output_tokens_est": int(generate.get("generation_output_tokens_est", 0) or 0),
        "generation_total_tokens_est": int(generate.get("generation_total_tokens_est", 0) or 0),
        "evidence_source_utilization_rate": float(generate.get("evidence_source_utilization_rate", 0.0) or 0.0),
        "context_budget_target_chunks": int(retrieve.get("context_budget_target_chunks", 0) or 0),
        "context_budget_policy": str(retrieve.get("context_budget_policy", "")),
        "corpus_scale": str(retrieve.get("corpus_scale", "")),
        "retrieval_top_k": int(retrieve.get("retrieval_top_k", 0) or 0),
        "retrieval_confidence": float(retrieve.get("retrieval_confidence", 0.0) or 0.0),
        "retrieval_score_gap": float(retrieve.get("retrieval_score_gap", 0.0) or 0.0),
        "evidence_compression_used": bool(retrieve.get("evidence_compression_used", False)),
        "evidence_compression_reduction_pct": float(retrieve.get("evidence_compression_reduction_pct", 0.0) or 0.0),
        "evidence_tokens_est_after_compression": int(retrieve.get("evidence_tokens_est_after_compression", 0) or 0),
    }


def _trace_node_times(trace: dict[str, Any], pacing_wait_ms: float = 0.0) -> dict[str, float]:
    """Return approximate service-node time with deliberate eval pacing removed.

    RequestPacer sleeps happen inside the LLM node that is about to issue a
    provider request. The trace records node wall time, so evaluation subtracts
    the query-level deliberate pacing proportionally across nodes according to
    their recorded LLM-call counts. Raw trace JSON remains unchanged.
    """
    nodes = list(trace.get("nodes", []))
    total_calls = sum(int(node.get("llm_calls", 0) or 0) for node in nodes)
    wait_per_call = (max(0.0, float(pacing_wait_ms)) / total_calls) if total_calls else 0.0
    out: dict[str, float] = {}
    for node in nodes:
        name = str(node.get("node", ""))
        if not name:
            continue
        raw = float(node.get("ms", 0.0) or 0.0)
        node_wait = wait_per_call * int(node.get("llm_calls", 0) or 0)
        service = max(0.0, raw - node_wait)
        out[name] = out.get(name, 0.0) + service
    return out


def _chunk_rank_metrics(hits: list[Any], case: dict[str, Any]) -> tuple[float | None, float | None]:
    terms = [str(x).lower() for x in case.get("chunk_must_contain", [])]
    if not terms:
        return None, None
    first_rank = 0
    for rank, hit in enumerate(hits[:5], start=1):
        text = hit.chunk.text.lower()
        if all(term in text for term in terms):
            first_rank = rank
            break
    return (float(first_rank == 1), (1.0 / first_rank if first_rank else 0.0))


def _node_latency_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        for node, ms in (row.get("_node_times") or {}).items():
            buckets.setdefault(node, []).append(float(ms))
    out = []
    for node, values in buckets.items():
        out.append({
            "node": node,
            "mean_ms": round(mean(values), 1),
            "p50_ms": round(percentile(values, 0.50), 1),
            "p95_ms": round(percentile(values, 0.95), 1),
            "samples": len(values),
        })
    return sorted(out, key=lambda row: float(row["mean_ms"]), reverse=True)


def _retrieval_ablation(workspace: Workspace, qa_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for use_reranker in (False, True):
        metric_buckets: dict[str, list[float]] = {
            "source_recall@5": [],
            "source_hit@1": [],
            "source_mrr": [],
            "source_ap@5": [],
            "source_ndcg@5": [],
            "source_duplicate_rate@5": [],
            "chunk_hit@1": [],
            "chunk_mrr": [],
        }
        latencies: list[float] = []
        for case in qa_cases:
            started = time.perf_counter()
            hits = workspace.retriever.search(case["question"], top_k=5, use_reranker=use_reranker)
            latency = (time.perf_counter() - started) * 1000
            returned = [hit.chunk.source for hit in hits]
            metrics = source_metrics(returned, case.get("relevant_sources", []))
            chunk_hit, chunk_mrr = _chunk_rank_metrics(hits, case)
            if chunk_hit is not None:
                metrics["chunk_hit@1"] = chunk_hit
                metrics["chunk_mrr"] = chunk_mrr
            for key in metric_buckets:
                if key in metrics:
                    metric_buckets[key].append(float(metrics[key]))
            latencies.append(latency)
        row = {
            "configuration": "Hybrid + reranker" if use_reranker else "Hybrid RRF",
            **{key: round(mean(values), 3) for key, values in metric_buckets.items()},
            "median_retrieval_ms": round(statistics.median(latencies), 1) if latencies else 0.0,
        }
        rows.append(row)
    return rows



def _context_budget_ablation(workspace: Workspace, qa_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Zero-Gemini comparison of full, fixed, and adaptive context budgets."""
    labels = ("Full top-k", "Fixed 3-chunk budget", "Adaptive budget")
    buckets: dict[str, list[dict[str, float]]] = {label: [] for label in labels}
    for case in qa_cases:
        plan = QueryPlan(
            route="documents",
            knowledge_scope="corpus",
            task_type="fact_lookup",
            retrieval_strategy="semantic",
            web_relevance="irrelevant",
            rewritten_query=case["question"],
            document_queries=[case["question"]],
        )
        cfg = PipelineConfig(profile="Balanced", top_k=6, use_context_pruning=True, use_adaptive_top_k=True)
        effective_k = adaptive_retrieval_top_k(
            cfg, plan, corpus_chunks=len(workspace.chunks), corpus_sources=len(workspace.source_profiles)
        )
        hits = workspace.retriever.search(case["question"], top_k=effective_k, use_reranker=False)
        fixed = focused_context_budget(hits, plan, cfg)
        adaptive = adaptive_context_budget(
            hits, plan, cfg, corpus_chunks=len(workspace.chunks), corpus_sources=len(workspace.source_profiles)
        )
        variants = {
            "Full top-k": (hits, len(hits), "full"),
            "Fixed 3-chunk budget": (fixed.hits, fixed.target_chunks, fixed.reason),
            "Adaptive budget": (adaptive.hits, adaptive.target_chunks, adaptive.reason),
        }
        full_chars = max(1, sum(len(hit.chunk.text or "") + len(hit.chunk.source or "") + 24 for hit in hits))
        for label, (variant, target, reason) in variants.items():
            returned = [hit.chunk.source for hit in variant]
            metrics = source_metrics(returned, case.get("relevant_sources", []))
            chars = sum(len(hit.chunk.text or "") + len(hit.chunk.source or "") + 24 for hit in variant)
            buckets[label].append({
                "source_precision@5": float(metrics["source_precision@5"]),
                "source_recall@5": float(metrics["source_recall@5"]),
                "source_hit@1": float(metrics["source_hit@1"]),
                "source_mrr": float(metrics["source_mrr"]),
                "context_chunks": float(len(variant)),
                "target_chunks": float(target),
                "context_sources": float(len(set(returned))),
                "context_chars": float(chars),
                "context_tokens_est": float((chars + 3) // 4),
                "context_reduction_pct": float(max(0.0, 1.0 - chars / full_chars) * 100.0),
                "adaptive_used": float(label == "Adaptive budget" and len(variant) < len(hits)),
            })

    rows: list[dict[str, Any]] = []
    for label in labels:
        values = buckets[label]
        rows.append({
            "configuration": label,
            "source_precision@5": round(mean([v["source_precision@5"] for v in values]), 3),
            "source_recall@5": round(mean([v["source_recall@5"] for v in values]), 3),
            "source_hit@1": round(mean([v["source_hit@1"] for v in values]), 3),
            "source_mrr": round(mean([v["source_mrr"] for v in values]), 3),
            "median_context_chunks": round(statistics.median([v["context_chunks"] for v in values]), 1) if values else 0.0,
            "median_target_chunks": round(statistics.median([v["target_chunks"] for v in values]), 1) if values else 0.0,
            "median_context_sources": round(statistics.median([v["context_sources"] for v in values]), 1) if values else 0.0,
            "median_context_chars": round(statistics.median([v["context_chars"] for v in values]), 1) if values else 0.0,
            "median_context_tokens_est": round(statistics.median([v["context_tokens_est"] for v in values]), 1) if values else 0.0,
            "median_context_reduction_pct": round(statistics.median([v["context_reduction_pct"] for v in values]), 1) if values else 0.0,
        })
    return rows


def _compression_signal(case: dict[str, Any], text: str) -> bool:
    lower = (text or "").lower()
    required = [str(term).lower() for term in case.get("chunk_must_contain", [])]
    if required:
        return all(term in lower for term in required)
    expected = [str(term).lower() for term in case.get("expected_any", [])]
    return any(term in lower for term in expected) if expected else True


def _evidence_compression_ablation(workspace: Workspace, qa_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Zero-Gemini test that focused sentence compression retains labeled answer signals."""
    full_tokens: list[float] = []
    compressed_tokens: list[float] = []
    retention: list[float] = []
    reductions: list[float] = []
    for case in qa_cases:
        plan = QueryPlan(
            route="documents", knowledge_scope="corpus", task_type="fact_lookup", retrieval_strategy="semantic",
            web_relevance="irrelevant", rewritten_query=case["question"], document_queries=[case["question"]],
        )
        cfg = PipelineConfig(profile="Balanced", top_k=6, use_context_pruning=True, use_adaptive_top_k=True)
        effective_k = adaptive_retrieval_top_k(
            cfg, plan, corpus_chunks=len(workspace.chunks), corpus_sources=len(workspace.source_profiles)
        )
        hits = workspace.retriever.search(case["question"], top_k=effective_k, use_reranker=False)
        budget = adaptive_context_budget(
            hits, plan, cfg, corpus_chunks=len(workspace.chunks), corpus_sources=len(workspace.source_profiles)
        )
        compression = focused_evidence_compression(budget.hits, plan, query=case["question"], enabled=True)
        before_text = "\n".join(hit.chunk.text or "" for hit in budget.hits)
        after_text = "\n".join(compression.texts.get(hit.chunk.id, hit.chunk.text or "") for hit in budget.hits)
        full_tokens.append(float((len(before_text) + 3) // 4))
        compressed_tokens.append(float((len(after_text) + 3) // 4))
        retention.append(float(_compression_signal(case, after_text)))
        reductions.append(float(compression.reduction_ratio * 100.0))

    return [
        {
            "configuration": "Adaptive context only",
            "answer_signal_retention": 1.0,
            "median_evidence_tokens_est": round(statistics.median(full_tokens), 1) if full_tokens else 0.0,
            "median_additional_reduction_pct": 0.0,
            "cases": len(qa_cases),
        },
        {
            "configuration": "Adaptive + sentence compression",
            "answer_signal_retention": round(mean(retention), 3),
            "median_evidence_tokens_est": round(statistics.median(compressed_tokens), 1) if compressed_tokens else 0.0,
            "median_additional_reduction_pct": round(statistics.median(reductions), 1) if reductions else 0.0,
            "cases": len(qa_cases),
        },
    ]


def _readiness_rows(summary: dict[str, Any], scale_rows: list[dict[str, Any]], compression_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    level = str(summary.get("evaluation_level", "Standard"))
    largest_scale = scale_rows[-1] if scale_rows else {}
    compressed = compression_rows[-1] if compression_rows else {}
    checks = [
        ("Answer accuracy", float(summary.get("answer_accuracy", 0.0)), 0.95, True),
        ("Source recall", float(summary.get("source_recall@5", 0.0)), 0.95, True),
        ("Citation validity", float(summary.get("citation_validity", 0.0)), 0.95, True),
        ("Citation coverage", float(summary.get("citation_coverage", 0.0)), 0.90, True),
        ("Planner route", float(summary.get("planner_route_accuracy", 0.0)), 0.90, True),
        ("Planner task", float(summary.get("planner_task_accuracy", 0.0)), 0.90, True),
        ("Planner strategy", float(summary.get("planner_strategy_accuracy", 0.0)), 0.90, True),
        ("Web precision", float(summary.get("web_use_precision", 0.0)), 0.90, True),
        ("Text2SQL", float(summary.get("text2sql_pass_rate", 0.0)), 0.90, True),
        ("Hard mode", float(summary.get("hard_mode_pass_rate", 0.0)), 0.85, True),
        ("Corpus overview", float(summary.get("overview_pass_rate", 0.0)), 0.90, True),
    ]
    # Quick intentionally skips local ablations and scale stress. Do not award
    # implicit PASS results for tests that were not executed. Standard/Deep
    # require them and missing rows therefore fail the corresponding gate.
    if level != "Quick":
        checks.extend([
            ("Adaptive-budget recall", float(summary.get("context_pruning_recall@5", 0.0)), 0.95, True),
            ("Compression signal retention", float(compressed.get("answer_signal_retention", 0.0)), 0.95, True),
            ("20x stress recall", float(largest_scale.get("source_recall@5", 0.0)), 0.95, True),
            ("20x stress pruned recall", float(largest_scale.get("adaptive_pruned_recall@5", 0.0)), 0.95, True),
        ])
    rows = []
    for name, value, threshold, critical in checks:
        rows.append({
            "check": name,
            "value": round(value, 3),
            "threshold": threshold,
            "status": "PASS" if value >= threshold else "FAIL",
            "critical": critical,
        })
    return rows


def _readiness_summary(rows: list[dict[str, Any]]) -> tuple[str, float]:
    critical = [row for row in rows if row.get("critical")]
    passed = sum(1 for row in critical if row.get("status") == "PASS")
    score = safe_div(passed, len(critical)) if critical else 0.0
    if score >= 1.0:
        return "READY", score
    if score >= 0.85:
        return "WATCH", score
    return "NOT READY", score


def _planner_eval(
    workspace: Workspace,
    cases: list[dict[str, Any]],
    gateway: GeminiGateway,
    progress: Callable[[float, str], None],
    start: float,
    span: float,
) -> list[dict[str, Any]]:
    manifest = workspace.manifest()
    rows: list[dict[str, Any]] = []
    total = max(1, len(cases))
    for idx, case in enumerate(cases, start=1):
        progress(start + span * (idx - 1) / total, f"Planner case {idx}/{len(cases)}")
        wait_before = gateway.request_pacer.total_sleep_seconds() if gateway.request_pacer else 0.0
        began = time.perf_counter()
        plan = gateway.analyze_query(case["question"], manifest, history=None, profile="Balanced")
        wall_latency = (time.perf_counter() - began) * 1000
        wait_after = gateway.request_pacer.total_sleep_seconds() if gateway.request_pacer else wait_before
        pacing_wait = max(0.0, wait_after - wait_before) * 1000
        latency = max(0.0, wall_latency - pacing_wait)
        planned_web = plan.web_relevance != "irrelevant" or plan.route in {"web", "hybrid"}
        rows.append(
            {
                "id": case["id"],
                "question": case["question"],
                "expected_route": case["route"],
                "route": plan.route,
                "route_correct": plan.route == case["route"],
                "expected_task": case["task"],
                "task": plan.task_type,
                "task_correct": plan.task_type == case["task"],
                "expected_strategy": case["strategy"],
                "strategy": plan.retrieval_strategy,
                "strategy_correct": plan.retrieval_strategy in set(case.get("strategy_any", [case["strategy"]])),
                "expected_web": bool(case["web_expected"]),
                "planned_web": planned_web,
                "latency_ms": round(latency, 1),
                "wall_latency_ms": round(wall_latency, 1),
                "pacing_wait_ms": round(pacing_wait, 1),
            }
        )
    return rows


def _judge_row(
    judge: GeminiGateway,
    case: dict[str, Any],
    answer: str,
    sources: list[dict[str, Any]],
    citations: dict[str, float | int],
) -> dict[str, Any]:
    wait_before = judge.request_pacer.total_sleep_seconds() if judge.request_pacer else 0.0
    began = time.perf_counter()
    judgement = judge.evaluate_rag_answer(
        case["question"],
        answer,
        _evidence_text(sources),
        case.get("reference_answer", ""),
        citation_validity=float(citations["citation_validity"]),
        citation_coverage=float(citations["citation_coverage"]),
    )
    judge_wall_latency = (time.perf_counter() - began) * 1000
    wait_after = judge.request_pacer.total_sleep_seconds() if judge.request_pacer else wait_before
    judge_pacing_wait = max(0.0, wait_after - wait_before) * 1000
    judge_latency = max(0.0, judge_wall_latency - judge_pacing_wait)
    return {
        "judge_faithfulness": round(judgement.faithfulness, 3),
        "judge_answer_relevance": round(judgement.answer_relevance, 3),
        "judge_completeness": round(judgement.completeness, 3),
        "judge_citation_support": round(judgement.citation_support, 3),
        "judge_overall": round(judgement.overall, 3),
        "judge_pass": judgement.pass_,
        "judge_reason": judgement.reason,
        "judge_latency_ms": round(judge_latency, 1),
        "judge_wall_latency_ms": round(judge_wall_latency, 1),
        "judge_pacing_wait_ms": round(judge_pacing_wait, 1),
    }


def _qa_eval(
    workspace: Workspace,
    cases: list[dict[str, Any]],
    api_key: str | None,
    model: str,
    deep_judge: bool,
    request_pacer: RequestPacer,
    progress: Callable[[float, str], None],
    start: float,
    span: float,
) -> list[dict[str, Any]]:
    engine = RAGEngine(workspace, request_pacer=request_pacer)
    judge = GeminiGateway(api_key, model, request_pacer=request_pacer) if deep_judge else None
    cfg = PipelineConfig(
        mode="Documents",
        profile="Fast",
        model=model,
        use_crag=False,
        allow_web_fallback=False,
        use_self_rag=False,
    )
    rows: list[dict[str, Any]] = []
    total = max(1, len(cases))
    for idx, case in enumerate(cases, start=1):
        progress(start + span * (idx - 1) / total, f"Document QA case {idx}/{len(cases)}")
        wait_before = request_pacer.total_sleep_seconds()
        began = time.perf_counter()
        result = engine.ask(case["question"], cfg, api_key, use_cache=False, record_history=False)
        wall_latency = (time.perf_counter() - began) * 1000
        pacing_wait = max(0.0, request_pacer.total_sleep_seconds() - wait_before) * 1000
        latency = max(0.0, wall_latency - pacing_wait)
        returned_sources = _document_sources(result.sources, 5)
        retrieval = source_metrics(returned_sources, case.get("relevant_sources", []))
        citations = citation_metrics(result.answer, result.sources)
        efficiency = _trace_efficiency(result.trace)
        row: dict[str, Any] = {
            "id": case["id"],
            "question": case["question"],
            "answer_key_match": answer_key_match(result.answer, case),
            **{key: round(float(value), 3) for key, value in retrieval.items()},
            "citation_count": citations["citation_count"],
            "citation_validity": round(float(citations["citation_validity"]), 3),
            "citation_coverage": round(float(citations["citation_coverage"]), 3),
            "confidence": round(result.confidence, 3),
            "latency_ms": round(latency, 1),
            "wall_latency_ms": round(wall_latency, 1),
            "pacing_wait_ms": round(pacing_wait, 1),
            **efficiency,
            "_answer": result.answer,
            "_sources": result.sources,
            "_citations": citations,
            "_node_times": _trace_node_times(result.trace, pacing_wait),
        }
        if judge and bool(case.get("deep_judge", False)):
            row.update(_judge_row(judge, case, result.answer, result.sources, citations))
        rows.append(row)
    return rows


def _overview_eval(
    workspace: Workspace,
    cases: list[dict[str, Any]],
    api_key: str | None,
    model: str,
    deep_judge: bool,
    request_pacer: RequestPacer,
    progress: Callable[[float, str], None],
    start: float,
    span: float,
) -> list[dict[str, Any]]:
    engine = RAGEngine(workspace, request_pacer=request_pacer)
    judge = GeminiGateway(api_key, model, request_pacer=request_pacer) if deep_judge else None
    cfg = PipelineConfig(
        mode="Auto",
        profile="Balanced",
        model=model,
        allow_web_fallback=True,
        use_crag=True,
        use_self_rag=False,
    )
    rows: list[dict[str, Any]] = []
    total = max(1, len(cases))
    for idx, case in enumerate(cases, start=1):
        progress(start + span * (idx - 1) / total, f"Corpus overview case {idx}/{len(cases)}")
        wait_before = request_pacer.total_sleep_seconds()
        began = time.perf_counter()
        result = engine.ask(case["question"], cfg, api_key, use_cache=False, record_history=False)
        wall_latency = (time.perf_counter() - began) * 1000
        pacing_wait = max(0.0, request_pacer.total_sleep_seconds() - wait_before) * 1000
        latency = max(0.0, wall_latency - pacing_wait)
        plan = result.trace.get("query_plan", {})
        evidence = result.trace.get("evidence", {})
        efficiency = _trace_efficiency(result.trace)
        coverage = float(evidence.get("source_coverage", 0.0) or 0.0)
        citations = citation_metrics(result.answer, result.sources)
        doc_sources = set(_document_sources(result.sources, 20))
        expected_task = case["expected_task"]
        expected_strategy = case["expected_strategy"]
        actual_task = plan.get("task_type")
        actual_strategy = plan.get("retrieval_strategy")
        # A generic collection summary can legitimately be made richer as an
        # insight synthesis. Treat overview/global and insight/analytical as the
        # same broad-collection family for this suite, while still requiring
        # local routing, breadth and no unnecessary web usage.
        task_ok = actual_task == expected_task or (
            expected_task == "overview" and actual_task == "insight_synthesis"
        )
        strategy_ok = actual_strategy == expected_strategy or (
            expected_task == "overview"
            and actual_task == "insight_synthesis"
            and actual_strategy == "analytical"
        )
        passed = (
            plan.get("route") == case["expected_route"]
            and task_ok
            and strategy_ok
            and efficiency["web_used"] == bool(case["web_expected"])
            and coverage >= float(case.get("min_source_coverage", 0.0))
            and float(citations.get("citation_validity", 0.0)) >= 0.80
        )
        row: dict[str, Any] = {
            "id": case["id"],
            "question": case["question"],
            "route": plan.get("route"),
            "task": plan.get("task_type"),
            "strategy": plan.get("retrieval_strategy"),
            "task_semantic_match": task_ok,
            "strategy_semantic_match": strategy_ok,
            "web_used": efficiency["web_used"],
            "source_coverage": round(coverage, 3),
            "document_sources_returned": len(doc_sources),
            "citation_validity": round(float(citations["citation_validity"]), 3),
            "citation_coverage": round(float(citations["citation_coverage"]), 3),
            "latency_ms": round(latency, 1),
            "wall_latency_ms": round(wall_latency, 1),
            "pacing_wait_ms": round(pacing_wait, 1),
            "pass": passed,
            **efficiency,
            "_answer": result.answer,
            "_sources": result.sources,
            "_citations": citations,
            "_node_times": _trace_node_times(result.trace, pacing_wait),
        }
        if judge and bool(case.get("deep_judge", False)):
            row.update(_judge_row(judge, case, result.answer, result.sources, citations))
        rows.append(row)
    return rows


def _sql_eval(
    workspace: Workspace,
    cases: list[dict[str, Any]],
    api_key: str | None,
    model: str,
    request_pacer: RequestPacer,
    progress: Callable[[float, str], None],
    start: float,
    span: float,
) -> list[dict[str, Any]]:
    """Evaluate Text2SQL generation/execution with one model call per case.

    SQL routing is already measured in the semantic-planner benchmark. Keeping
    this component test route-independent avoids spending two extra Gemini
    calls per case just to duplicate planner and answer-generation coverage.
    """
    gateway = GeminiGateway(api_key, model, request_pacer=request_pacer)
    rows: list[dict[str, Any]] = []
    total = max(1, len(cases))
    for idx, case in enumerate(cases, start=1):
        progress(start + span * (idx - 1) / total, f"Text2SQL case {idx}/{len(cases)}")
        wait_before = request_pacer.total_sleep_seconds()
        began = time.perf_counter()
        try:
            sql, result = workspace.sql.benchmark_query(case["question"], gateway)
            preview = result.head(200)
            result_text = preview.to_markdown(index=False) if len(preview) else "(no rows)"
            observed_scalar = preview.iloc[0, 0] if len(preview) and len(preview.columns) else None
            if "expected_scalar" in case:
                matched = scalar_value_match(observed_scalar, case.get("expected_scalar"))
                match_method = "typed_scalar"
            else:
                matched = answer_key_match(result_text, case)
                match_method = "rendered_answer_key"
            error = ""
            readonly_validated = True
        except Exception as exc:
            sql = ""
            result = None
            result_text = ""
            matched = False
            match_method = "error"
            observed_scalar = None
            error = f"{type(exc).__name__}: {exc}"
            readonly_validated = False
        wall_latency = (time.perf_counter() - began) * 1000
        pacing_wait = max(0.0, request_pacer.total_sleep_seconds() - wait_before) * 1000
        latency = max(0.0, wall_latency - pacing_wait)
        rows.append(
            {
                "id": case["id"],
                "question": case["question"],
                "component": "Text2SQL",
                "answer_key_match": matched,
                "match_method": match_method,
                "observed_value": (
                    observed_scalar.item() if hasattr(observed_scalar, "item") else observed_scalar
                ),
                "expected_value": case.get("expected_scalar", ""),
                "readonly_validated": readonly_validated,
                "sql": sql,
                "rows": int(len(result)) if result is not None else 0,
                "latency_ms": round(latency, 1),
                "wall_latency_ms": round(wall_latency, 1),
                "pacing_wait_ms": round(pacing_wait, 1),
                "llm_calls_estimate": 1,
                "error": error,
            }
        )
    return rows


def _abstention_eval() -> list[dict[str, Any]]:
    empty = Workspace(f"eval-empty-{uuid.uuid4().hex[:10]}")
    engine = RAGEngine(empty)
    cases = [
        ("Documents", "What are the documents about?", "workspace_empty_documents"),
        ("Data (SQL)", "Which row has the highest value?", "workspace_empty_tables"),
    ]
    rows = []
    for mode, question, expected_reason in cases:
        result = engine.ask(
            question,
            PipelineConfig(mode=mode, profile="Fast"),
            api_key=None,
            use_cache=False,
            record_history=False,
        )
        nodes = result.trace.get("nodes", [])
        abstain_node = next((node for node in nodes if node.get("node") == "abstain"), {})
        rows.append(
            {
                "mode": mode,
                "question": question,
                "abstained": bool(abstain_node),
                "reason": abstain_node.get("reason"),
                "expected_reason": expected_reason,
                "pass": abstain_node.get("reason") == expected_reason,
                "llm_calls_estimate": result.trace.get("metrics", {}).get("llm_calls_estimate", 0),
            }
        )
    return rows


def _planner_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    tp = fp = fn = 0
    for row in rows:
        expected = bool(row["expected_web"])
        planned = bool(row["planned_web"])
        if expected and planned:
            tp += 1
        elif not expected and planned:
            fp += 1
        elif expected and not planned:
            fn += 1
    return {
        "planner_route_accuracy": mean([float(row["route_correct"]) for row in rows]),
        "planner_task_accuracy": mean([float(row["task_correct"]) for row in rows]),
        "planner_strategy_accuracy": mean([float(row["strategy_correct"]) for row in rows]),
        "web_use_precision": safe_div(tp, tp + fp),
        "web_use_recall": safe_div(tp, tp + fn),
        "unnecessary_web_rate": safe_div(fp, sum(1 for row in rows if not row["expected_web"])),
    }


def _base_grade(score: float) -> str:
    if score >= 0.90:
        return "A"
    if score >= 0.80:
        return "B"
    if score >= 0.70:
        return "C"
    if score >= 0.60:
        return "D"
    return "Needs work"


def _grade_with_gates(score: float, metrics: dict[str, float]) -> tuple[str, list[str]]:
    """Prevent a weighted average from hiding a badly failing subsystem."""
    order = ["Needs work", "D", "C", "B", "A"]
    grade = _base_grade(score)
    gates: list[str] = []

    def cap(max_grade: str, reason: str) -> None:
        nonlocal grade
        if order.index(grade) > order.index(max_grade):
            grade = max_grade
        gates.append(reason)

    if metrics["planner_route_accuracy"] < 0.90 or metrics["web_use_precision"] < 0.90:
        cap("C", "Critical routing/web-policy accuracy is below 90%.")
    if metrics["text2sql_pass_rate"] < 0.75:
        cap("B", "Text2SQL pass rate is below 75%.")
    if metrics["citation_validity"] < 0.90:
        cap("B", "Citation validity is below 90%.")
    if metrics["citation_coverage"] < 0.80:
        cap("B", "Citation coverage is below 80%.")
    if metrics["planner_task_accuracy"] < 0.75:
        cap("B", "Planner task taxonomy accuracy is below 75%.")
    if metrics.get("hard_mode_pass_rate", 1.0) < 0.50:
        cap("C", "Hard-mode robustness pass rate is below 50%.")
    elif metrics.get("hard_mode_pass_rate", 1.0) < 0.75:
        cap("B", "Hard-mode robustness pass rate is below 75%.")
    return grade, gates


def _hard_mode_eval(
    workspace: Workspace,
    cases: list[dict[str, Any]],
    api_key: str | None,
    model: str,
    request_pacer: RequestPacer,
    progress: Callable[[float, str], None],
    start: float,
    span: float,
) -> list[dict[str, Any]]:
    engine = RAGEngine(workspace, request_pacer=request_pacer)
    gateway = GeminiGateway(api_key, model, request_pacer=request_pacer)
    rows: list[dict[str, Any]] = []
    total = max(1, len(cases))
    for idx, case in enumerate(cases, start=1):
        progress(start + span * (idx - 1) / total, f"Hard-mode case {idx}/{len(cases)}")
        kind = case.get("kind", "qa")
        row: dict[str, Any] = {"id": case["id"], "kind": kind, "question": case.get("question", "")}
        if kind == "security":
            score = prompt_injection_score(case.get("text", ""))
            row.update({"injection_score": round(score, 3), "pass": score >= float(case.get("min_injection_score", 0.5)), "gemini_calls": 0})
            rows.append(row)
            continue
        if kind == "sql":
            wait_before = request_pacer.total_sleep_seconds()
            began = time.perf_counter()
            try:
                sql, result = workspace.sql.benchmark_query(case["question"], gateway)
                observed = result.iloc[0, 0] if len(result) and len(result.columns) else None
                passed = scalar_value_match(observed, case.get("expected_scalar"))
                error = ""
            except Exception as exc:
                sql, observed, passed = "", None, False
                error = f"{type(exc).__name__}: {exc}"
            wall = (time.perf_counter() - began) * 1000
            pace = max(0.0, request_pacer.total_sleep_seconds() - wait_before) * 1000
            row.update({"observed_value": observed.item() if hasattr(observed, "item") else observed, "expected_value": case.get("expected_scalar"), "sql": sql, "pass": passed, "latency_ms": round(max(0.0, wall-pace),1), "error": error, "gemini_calls": 1})
            rows.append(row)
            continue
        if kind == "planner":
            wait_before = request_pacer.total_sleep_seconds()
            began = time.perf_counter()
            plan = gateway.analyze_query(case["question"], workspace.manifest(), history=None, profile="Balanced")
            wall = (time.perf_counter() - began) * 1000
            pace = max(0.0, request_pacer.total_sleep_seconds() - wait_before) * 1000
            passed = plan.route == case["route"] and plan.task_type == case["task"] and plan.retrieval_strategy == case["strategy"] and ((plan.web_relevance != "irrelevant") == bool(case["web_expected"]))
            row.update({"route":plan.route,"task":plan.task_type,"strategy":plan.retrieval_strategy,"web_relevance":plan.web_relevance,"pass":passed,"latency_ms":round(max(0.0,wall-pace),1),"gemini_calls":1})
            rows.append(row)
            continue

        case_mode = str(case.get("mode") or "Documents")
        case_profile = str(case.get("profile") or "Fast")
        cfg = PipelineConfig(
            mode=case_mode,
            profile=case_profile,
            model=model,
            use_crag=bool(case.get("use_crag", False)),
            allow_web_fallback=False,
            use_self_rag=False,
        )
        if kind == "insight":
            cfg = PipelineConfig(mode="Auto", profile="Balanced", model=model, use_crag=True, allow_web_fallback=False, use_self_rag=False)
        wait_before = request_pacer.total_sleep_seconds()
        began = time.perf_counter()
        result = engine.ask(case["question"], cfg, api_key, use_cache=False, record_history=False)
        wall = (time.perf_counter() - began) * 1000
        pace = max(0.0, request_pacer.total_sleep_seconds() - wait_before) * 1000
        plan = result.trace.get("query_plan", {})
        citations = citation_metrics(result.answer, result.sources)
        returned = _document_sources(result.sources, 5)
        retrieval = source_metrics(returned, case.get("relevant_sources", [])) if case.get("relevant_sources") else {}
        if kind == "missing":
            grounded_absence = bool(result.trace.get("metrics", {}).get("grounded_absence", False))
            passed = missing_answer_match(result.answer, case) or grounded_absence
        elif kind == "insight":
            table_cited = any(str(src.get("id", "")).startswith("T") for src in result.sources) and "[T" in result.answer
            evidence = result.trace.get("evidence", {})
            passed = plan.get("task_type") == case.get("expected_task") and plan.get("retrieval_strategy") == case.get("expected_strategy") and float(evidence.get("source_coverage", 0.0) or 0.0) >= float(case.get("min_source_coverage", 0.0)) and (table_cited or not case.get("requires_table_citation"))
        else:
            passed = answer_key_match(result.answer, case) and float(retrieval.get("source_recall@5", 1.0)) >= 1.0
            expected_route = case.get("expected_route")
            expected_task = case.get("expected_task")
            expected_strategy = case.get("expected_strategy")
            if expected_route:
                passed = passed and plan.get("route") == expected_route
            if expected_task:
                passed = passed and plan.get("task_type") == expected_task
            if expected_strategy:
                passed = passed and plan.get("retrieval_strategy") == expected_strategy
        row.update({
            "evaluation_mode": cfg.mode if kind not in {"sql", "planner", "security"} else None,
            "evaluation_profile": cfg.profile if kind not in {"sql", "planner", "security"} else None,
            "route": plan.get("route"),
            "task": plan.get("task_type"),
            "strategy": plan.get("retrieval_strategy"),
            "answer_key_match": answer_key_match(result.answer, case) if kind == "qa" else None,
            "missing_answer_match": missing_answer_match(result.answer, case) if kind == "missing" else None,
            "grounded_absence": bool(result.trace.get("metrics", {}).get("grounded_absence", False)) if kind == "missing" else None,
            "citation_validity": round(float(citations["citation_validity"]), 3),
            "citation_coverage": round(float(citations["citation_coverage"]), 3),
            "source_recall@5": round(float(retrieval.get("source_recall@5", 1.0)), 3),
            "latency_ms": round(max(0.0, wall - pace), 1),
            "pass": passed,
            "gemini_calls": int(result.trace.get("metrics", {}).get("llm_calls_estimate", 0) or 0),
            "_answer": result.answer,
            "_sources": result.sources,
            "_node_times": _trace_node_times(result.trace, pace),
        })
        rows.append(row)
    return rows


def _profile_benchmark(
    workspace: Workspace,
    cases: list[dict[str, Any]],
    api_key: str | None,
    model: str,
    request_pacer: RequestPacer,
    progress: Callable[[float, str], None],
    start: float,
    span: float,
) -> list[dict[str, Any]]:
    selected = [cases[0]] if cases else []
    cross = next((case for case in cases if len(case.get("relevant_sources", [])) > 1), None)
    if cross and cross not in selected:
        selected.append(cross)
    elif len(cases) > 1:
        selected.append(cases[1])
    rows: list[dict[str, Any]] = []
    combinations = [(profile, case) for profile in ("Fast", "Balanced", "Agentic") for case in selected]
    total = max(1, len(combinations))
    for idx, (profile, case) in enumerate(combinations, start=1):
        progress(
            start + span * (idx - 1) / total,
            f"Profile benchmark {idx}/{len(combinations)}",
        )

        cfg = PipelineConfig(
            mode="Documents",
            profile=profile,
            model=model,
            allow_web_fallback=False,
            use_crag=True,
            use_self_rag=True,
        )
        engine = RAGEngine(workspace, request_pacer=request_pacer)

        wait_before = request_pacer.total_sleep_seconds()
        began = time.perf_counter()

        result = engine.ask(
            case["question"],
            cfg,
            api_key,
            use_cache=False,
            record_history=False,
        )

        wall = (time.perf_counter() - began) * 1000
        pace = max(
            0.0,
            request_pacer.total_sleep_seconds() - wait_before,
        ) * 1000

        citations = citation_metrics(result.answer, result.sources)
        metrics = result.trace.get("metrics", {})

        rows.append(
            {
                "profile": profile,
                "case": case["id"],
                "answer_key_match": answer_key_match(result.answer, case),
                "citation_validity": round(float(citations["citation_validity"]), 3),
                "citation_coverage": round(float(citations["citation_coverage"]), 3),
                "latency_ms": round(max(0.0, wall - pace), 1),
                "llm_calls_estimate": int(metrics.get("llm_calls_estimate", 0) or 0),
                "reranker_used": bool(metrics.get("reranker_used", False)),
                "correction_used": bool(metrics.get("correction_used", False)),
            }
        )
        return rows


def _profile_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for profile in ("Fast", "Balanced", "Agentic"):
        group = [row for row in rows if row.get("profile") == profile]
        if not group:
            continue
        summaries.append({
            "profile": profile,
            "answer_accuracy": round(mean([float(bool(row.get("answer_key_match"))) for row in group]), 3),
            "citation_validity": round(mean([float(row.get("citation_validity", 0.0) or 0.0) for row in group]), 3),
            "citation_coverage": round(mean([float(row.get("citation_coverage", 0.0) or 0.0) for row in group]), 3),
            "median_latency_ms": round(percentile([float(row.get("latency_ms", 0.0) or 0.0) for row in group], 0.5), 1),
            "mean_llm_calls": round(mean([float(row.get("llm_calls_estimate", 0.0) or 0.0) for row in group]), 2),
            "reranker_rate": round(mean([float(bool(row.get("reranker_used"))) for row in group]), 3),
            "cases": len(group),
        })
    return summaries


def _profile_recommendation(profile_summary: list[dict[str, Any]]) -> str:
    if not profile_summary:
        return ""
    by_name = {row["profile"]: row for row in profile_summary}
    fast = by_name.get("Fast")
    balanced = by_name.get("Balanced")
    agentic = by_name.get("Agentic")
    if fast and balanced:
        quality_close = (
            float(fast.get("answer_accuracy", 0.0)) >= float(balanced.get("answer_accuracy", 0.0)) - 0.01
            and float(fast.get("citation_coverage", 0.0)) >= float(balanced.get("citation_coverage", 0.0)) - 0.05
        )
        faster = float(fast.get("median_latency_ms", 0.0) or 0.0) < float(balanced.get("median_latency_ms", 0.0) or 0.0)
        if quality_close and faster:
            return "Fast matched Balanced quality on the sampled explicit-Documents cases with lower median latency. Keep Balanced as the general Auto default, but prefer Fast for simple local lookups."
    if agentic and balanced and float(agentic.get("median_latency_ms", 0.0) or 0.0) > 2 * max(1.0, float(balanced.get("median_latency_ms", 0.0) or 0.0)):
        return "Agentic was materially slower than Balanced on the sampled cases. Reserve Agentic for difficult or low-confidence work rather than routine lookups."
    return "Profile differences were not large enough on this sample to justify changing the default execution policy."


def _diagnostics(
    summary: dict[str, Any],
    ablation_rows: list[dict[str, Any]],
    planner_rows: list[dict[str, Any]],
    sql_rows: list[dict[str, Any]],
    hard_rows: list[dict[str, Any]] | None = None,
    profile_summary: list[dict[str, Any]] | None = None,
    node_latency_rows: list[dict[str, Any]] | None = None,
    context_budget_rows: list[dict[str, Any]] | None = None,
    compression_rows: list[dict[str, Any]] | None = None,
    scale_stress_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []

    if int(summary.get("rate_limit_retries", 0) or 0) > 0:
        findings.append(
            {
                "severity": "warning",
                "area": "gemini quota",
                "finding": (
                    f"Gemini surfaced {int(summary.get('rate_limit_retries', 0))} rate-limit retry event(s); "
                    f"provider-directed retry wait was {float(summary.get('rate_limit_sleep_ms', 0.0)) / 1000:.1f}s."
                ),
                "recommendation": "Keep quota-safe pacing enabled or lower the target RPM below the active project limit.",
            }
        )

    context_budget_rows = context_budget_rows or []
    adaptive_row = next(
        (row for row in context_budget_rows if row.get("configuration") == "Adaptive budget"), {}
    )
    full_row = next((row for row in context_budget_rows if row.get("configuration") == "Full top-k"), {})
    if summary.get("source_recall@5", 0.0) >= 0.95 and summary.get("source_precision@5", 1.0) < 0.60:
        if adaptive_row:
            findings.append({
                "severity": "info",
                "area": "context efficiency",
                "finding": (
                    f"Source Recall@5 is {float(summary.get('source_recall@5', 0.0)):.0%}; adaptive focused context now "
                    f"uses a median target of {float(adaptive_row.get('median_target_chunks', 0.0)):.1f} chunks and reduces "
                    f"context by {float(adaptive_row.get('median_context_reduction_pct', 0.0)):.0f}%."
                ),
                "recommendation": "The runtime already applies adaptive budgeting. Use the compression and scale-stress ablations to decide whether further tightening is safe rather than lowering top-k globally.",
            })
        else:
            findings.append({
                "severity": "info",
                "area": "context efficiency",
                "finding": (
                    f"Source Recall@5 is {float(summary.get('source_recall@5', 0.0)):.0%} while source Precision@5 is "
                    f"{float(summary.get('source_precision@5', 0.0)):.0%}."
                ),
                "recommendation": "Use a focused context budget before reducing global retrieval breadth.",
            })

    if adaptive_row and full_row:
        full_precision = float(full_row.get("source_precision@5", 0.0) or 0.0)
        adaptive_precision = float(adaptive_row.get("source_precision@5", 0.0) or 0.0)
        full_recall = float(full_row.get("source_recall@5", 0.0) or 0.0)
        adaptive_recall = float(adaptive_row.get("source_recall@5", 0.0) or 0.0)
        reduction = float(adaptive_row.get("median_context_reduction_pct", 0.0) or 0.0)
        if adaptive_recall >= full_recall - 1e-9 and reduction >= 25.0:
            findings.append({
                "severity": "ok",
                "area": "adaptive context budget",
                "finding": (
                    f"Adaptive budgeting preserved source Recall@5 at {adaptive_recall:.0%}, changed Precision@5 from "
                    f"{full_precision:.0%} to {adaptive_precision:.0%}, and cut median context by {reduction:.0f}%."
                ),
                "recommendation": "Keep adaptive budgeting enabled; corpus-scale stress now provides the guardrail for future budget changes.",
            })
        elif adaptive_recall < full_recall - 1e-9:
            findings.append({
                "severity": "warning",
                "area": "adaptive context budget",
                "finding": f"Adaptive budgeting reduced source Recall@5 from {full_recall:.0%} to {adaptive_recall:.0%}.",
                "recommendation": "Loosen the adaptive budget before shipping this policy broadly.",
            })

    compression_rows = compression_rows or []
    compressed = next(
        (row for row in compression_rows if row.get("configuration") == "Adaptive + sentence compression"), {}
    )
    if compressed:
        retention = float(compressed.get("answer_signal_retention", 0.0) or 0.0)
        reduction = float(compressed.get("median_additional_reduction_pct", 0.0) or 0.0)
        findings.append({
            "severity": "ok" if retention >= 0.95 else "warning",
            "area": "evidence compression",
            "finding": f"Focused sentence compression retained labeled answer signals in {retention:.0%} of cases while cutting selected-evidence tokens by a median {reduction:.0f}% beyond context budgeting.",
            "recommendation": "Keep compression enabled for focused lookups only." if retention >= 0.95 else "Disable or loosen sentence compression until labeled signal retention returns above 95%.",
        })

    scale_stress_rows = scale_stress_rows or []
    if summary.get("scale_stress_error"):
        findings.append({
            "severity": "warning",
            "area": "scale stress",
            "finding": f"The zero-Gemini scale-stress harness did not complete: {summary.get('scale_stress_error')}",
            "recommendation": "Treat release readiness as incomplete until the local scale-stress harness runs successfully; the main RAG benchmark can still be inspected independently.",
        })
    if scale_stress_rows:
        largest = scale_stress_rows[-1]
        recall = float(largest.get("source_recall@5", 0.0) or 0.0)
        pruned_recall = float(largest.get("adaptive_pruned_recall@5", 0.0) or 0.0)
        findings.append({
            "severity": "ok" if min(recall, pruned_recall) >= 0.95 else "warning",
            "area": "scale stress",
            "finding": (
                f"At {int(largest.get('chunks', 0) or 0)} chunks / {int(largest.get('sources', 0) or 0)} sources, "
                f"retrieval Recall@5 was {recall:.0%} and adaptive-pruned recall was {pruned_recall:.0%}."
            ),
            "recommendation": "Treat this as synthetic distractor evidence, then repeat with a real larger upload before changing the reranker policy." if min(recall, pruned_recall) >= 0.95 else "Increase retrieval depth or budget targets for large corpora before relying on the adaptive policy.",
        })

    if summary.get("citation_coverage", 1.0) < 0.90:
        findings.append(
            {
                "severity": "warning",
                "area": "citations",
                "finding": f"Citation coverage is {float(summary['citation_coverage']):.0%}; some factual statements are uncited.",
                "recommendation": "Keep the generation prompt citation requirement and inspect low-coverage cases individually.",
            }
        )
    if summary.get("planner_task_accuracy", 1.0) < 0.90:
        failed = [row["id"] for row in planner_rows if not row.get("task_correct")]
        findings.append(
            {
                "severity": "warning",
                "area": "planner",
                "finding": f"Task classification misses: {', '.join(failed) or 'none'}.",
                "recommendation": "Review task taxonomy labels separately from route/strategy correctness; do not over-penalize equivalent plans.",
            }
        )
    if summary.get("text2sql_pass_rate", 1.0) < 0.90:
        failed = [row["id"] for row in sql_rows if not row.get("answer_key_match")]
        findings.append(
            {
                "severity": "warning",
                "area": "text2sql",
                "finding": f"Text2SQL failed cases: {', '.join(failed) or 'none'}.",
                "recommendation": (
                    "Inspect generated SQL, typed observed values and benchmark expectations. SQL routing is evaluated "
                    "separately in the planner suite, so a component failure should not automatically be blamed on routing."
                ),
            }
        )

    hard_rows = hard_rows or []
    if hard_rows and summary.get("hard_mode_pass_rate", 1.0) < 0.90:
        failed = [row.get("id", "?") for row in hard_rows if not row.get("pass")]
        findings.append({
            "severity": "warning",
            "area": "hard_mode",
            "finding": f"Hard-mode robustness failures: {', '.join(failed) or 'none'}.",
            "recommendation": "Inspect missing-answer, distractor, analytical and adversarial cases before expanding the feature set.",
        })
    if len(ablation_rows) == 2:
        base, rerank = ablation_rows
        base_ms = float(base.get("median_retrieval_ms", 0.0) or 0.0)
        rerank_ms = float(rerank.get("median_retrieval_ms", 0.0) or 0.0)
        base_mrr = float(base.get("source_mrr", 0.0) or 0.0)
        rerank_mrr = float(rerank.get("source_mrr", 0.0) or 0.0)
        multiplier = safe_div(rerank_ms, base_ms) if base_ms else 0.0
        if multiplier >= 3.0 and rerank_mrr <= base_mrr + 0.01:
            findings.append(
                {
                    "severity": "info",
                    "area": "reranker",
                    "finding": f"The reranker ablation is {multiplier:.1f}x slower on the demo benchmark with no material source-MRR gain.",
                    "recommendation": "The runtime already skips reranking on small corpora. Keep this ablation as evidence and re-enable the cross-encoder only when a larger-corpus benchmark shows source- or chunk-level gain.",
                }
            )
    profile_summary = profile_summary or []
    recommendation = _profile_recommendation(profile_summary)
    if recommendation:
        findings.append({
            "severity": "info",
            "area": "profile policy",
            "finding": recommendation,
            "recommendation": "Use the profile benchmark as local evidence only; repeat it on larger user corpora before making a global policy claim.",
        })

    node_latency_rows = node_latency_rows or []
    if node_latency_rows:
        dominant = max(node_latency_rows, key=lambda row: float(row.get("mean_ms", 0.0) or 0.0))
        total_mean = sum(float(row.get("mean_ms", 0.0) or 0.0) for row in node_latency_rows)
        share = safe_div(float(dominant.get("mean_ms", 0.0) or 0.0), total_mean)
        if share >= 0.60:
            findings.append({
                "severity": "info",
                "area": "latency",
                "finding": f"{dominant.get('node', 'generation')} dominates mean node time at approximately {share:.0%} of measured pipeline-node latency.",
                "recommendation": "Prioritize model/generation efficiency before micro-optimizing millisecond-scale retrieval stages.",
            })

    if not findings:
        findings.append(
            {
                "severity": "ok",
                "area": "benchmark",
                "finding": "No configured quality gate produced a diagnostic warning.",
                "recommendation": "Expand the benchmark before treating this as general performance evidence.",
            }
        )
    return findings


def _deep_from_standard_cache(
    base_report: dict[str, Any],
    api_key: str | None,
    model: str,
    *,
    target_rpm: int,
    progress: Callable[[float, str], None],
) -> dict[str, Any]:
    """Upgrade a cached Standard run to Deep with judge calls only.

    The deterministic suites are identical between Standard and Deep. Reusing a
    current Standard baseline avoids spending ~25 repeated Gemini calls merely
    to regenerate metrics the user already computed. Deep then adds the sampled
    calibrated judge layer on top of those exact answers/evidence artifacts.
    """
    started = time.perf_counter()
    report = json.loads(json.dumps(base_report))
    benchmark = _load_benchmark()
    cases = {
        case["id"]: case
        for case in benchmark.get("qa_cases", []) + benchmark.get("overview_cases", [])
        if case.get("deep_judge")
    }
    request_pacer = RequestPacer(target_rpm=max(0, int(target_rpm)))
    judge = GeminiGateway(api_key, model, request_pacer=request_pacer)
    judge_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for section in ("focused_qa", "corpus_overviews"):
        for row in report.get(section, []):
            if row.get("id") in cases and row.get("_answer") is not None and row.get("_sources") is not None:
                candidate_rows.append(row)

    total = max(1, len(candidate_rows))
    for idx, row in enumerate(candidate_rows, start=1):
        progress(0.08 + 0.84 * (idx - 1) / total, f"Deep judge case {idx}/{len(candidate_rows)}")
        case = cases[row["id"]]
        citations = row.get("_citations") or citation_metrics(row.get("_answer", ""), row.get("_sources", []))
        row.update(_judge_row(judge, case, row.get("_answer", ""), row.get("_sources", []), citations))
        judge_rows.append(row)

    summary = report.setdefault("summary", {})
    baseline_wall_ms = float(summary.get("evaluation_wall_ms", 0.0) or 0.0)
    if judge_rows:
        summary.update(
            {
                "judge_faithfulness": round(mean([float(row["judge_faithfulness"]) for row in judge_rows]), 3),
                "judge_answer_relevance": round(mean([float(row["judge_answer_relevance"]) for row in judge_rows]), 3),
                "judge_completeness": round(mean([float(row["judge_completeness"]) for row in judge_rows]), 3),
                "judge_citation_support": round(mean([float(row["judge_citation_support"]) for row in judge_rows]), 3),
                "judge_overall": round(mean([float(row["judge_overall"]) for row in judge_rows]), 3),
                "judge_pass_rate": round(mean([float(row["judge_pass"]) for row in judge_rows]), 3),
                "judge_latency_mean_ms": round(mean([float(row.get("judge_latency_ms", 0.0)) for row in judge_rows]), 3),
            }
        )
    pacing_stats = request_pacer.stats()
    summary.update(
        {
            "evaluation_level": "Deep",
            "evaluation_wall_ms": round((time.perf_counter() - started) * 1000, 1),
            "evaluation_target_rpm": int(pacing_stats["target_rpm"]),
            "gemini_requests": int(pacing_stats["gemini_requests"]),
            "pacing_sleep_ms": float(pacing_stats["pacing_sleep_ms"]),
            "rate_limit_retries": int(pacing_stats["rate_limit_retries"]),
            "rate_limit_sleep_ms": float(pacing_stats["rate_limit_sleep_ms"]),
            "deep_judge_cases": len(judge_rows),
            "reused_standard_baseline": True,
            "deep_incremental": True,
            "deterministic_baseline_wall_ms": round(baseline_wall_ms, 1),
        }
    )
    report["diagnostics"] = _diagnostics(
        summary,
        report.get("retrieval_ablation", []),
        report.get("semantic_planner", []),
        report.get("text2sql", []),
        report.get("hard_mode", []),
        report.get("profile_summary", []),
        report.get("node_latency", []),
        report.get("context_budget_ablation", []),
        report.get("evidence_compression_ablation", []),
        report.get("scale_stress", []),
    )
    report.setdefault("methodology", {})["evaluation_cache"] = (
        "Deep reused the current cached Standard deterministic baseline and issued only sampled judge calls."
    )
    progress(1.0, "Deep evaluation complete")
    return report


def run_demo_eval(
    workspace: Workspace,
    api_key: str | None,
    model: str,
    level: str = "Standard",
    progress_callback: Callable[[float, str], None] | None = None,
    target_rpm: int = 12,
    base_standard_report: dict[str, Any] | None = None,
    include_profile_benchmark: bool = False,
) -> dict[str, Any]:
    """Run the bundled benchmark with response caching disabled.

    Quick: smaller deterministic regression set.
    Standard: full deterministic set + retrieval and context-budget ablations.
    Deep: Standard plus calibrated Gemini LLM-as-judge scores.
    """
    wall_started = time.perf_counter()
    progress = progress_callback or (lambda _value, _message: None)
    benchmark = _load_benchmark()
    level = level if level in {"Quick", "Standard", "Deep"} else "Standard"
    if level == "Deep" and base_standard_report and not include_profile_benchmark:
        artifact_rows = base_standard_report.get("focused_qa", []) + base_standard_report.get("corpus_overviews", [])
        if any(row.get("_answer") is not None and row.get("_sources") is not None for row in artifact_rows):
            return _deep_from_standard_cache(
                base_standard_report,
                api_key,
                model,
                target_rpm=target_rpm,
                progress=progress,
            )
    deep_judge = level == "Deep"
    request_pacer = RequestPacer(target_rpm=max(0, int(target_rpm)))

    qa_cases = benchmark["qa_cases"] if level != "Quick" else benchmark["qa_cases"][:3]
    planner_cases = benchmark["planner_cases"] if level != "Quick" else benchmark["planner_cases"][:5]
    overview_cases = benchmark["overview_cases"] if level != "Quick" else benchmark["overview_cases"][:1]
    sql_cases = benchmark.get("sql_cases", []) if level != "Quick" else benchmark.get("sql_cases", [])[:1]
    hard_cases = benchmark.get("hard_mode_cases", []) if level != "Quick" else benchmark.get("hard_mode_cases", [])[:2]

    progress(0.01, "Preparing evaluation")
    qa_rows = _qa_eval(
        workspace, qa_cases, api_key, model, deep_judge, request_pacer, progress, 0.03, 0.29
    )
    gateway = GeminiGateway(api_key, model, request_pacer=request_pacer)
    planner_rows = _planner_eval(workspace, planner_cases, gateway, progress, 0.34, 0.22)
    overview_rows = _overview_eval(
        workspace, overview_cases, api_key, model, deep_judge, request_pacer, progress, 0.58, 0.16
    )
    sql_rows = (
        _sql_eval(workspace, sql_cases, api_key, model, request_pacer, progress, 0.75, 0.08)
        if sql_cases
        else []
    )
    hard_rows = _hard_mode_eval(
        workspace, hard_cases, api_key, model, request_pacer, progress, 0.83, 0.10
    ) if hard_cases else []
    profile_rows = _profile_benchmark(
        workspace, qa_cases, api_key, model, request_pacer, progress, 0.93, 0.05
    ) if include_profile_benchmark and level != "Quick" else []
    progress(0.98, "Checking abstention, retrieval, context and scale ablations")
    abstention_rows = _abstention_eval()
    ablation_rows = _retrieval_ablation(workspace, qa_cases) if level != "Quick" else []
    context_budget_rows = _context_budget_ablation(workspace, qa_cases) if level != "Quick" else []
    compression_rows = _evidence_compression_ablation(workspace, qa_cases) if level != "Quick" else []
    scale_stress_error = ""
    if level != "Quick":
        try:
            scale_stress_rows = scale_stress_retrieval_eval(workspace, qa_cases)
        except Exception as exc:  # keep the primary benchmark available even if the local stress harness fails
            scale_stress_rows = []
            scale_stress_error = f"{type(exc).__name__}: {exc}"
    else:
        scale_stress_rows = []

    planner_metrics = _planner_summary(planner_rows)
    qa_latencies = [float(row["latency_ms"]) for row in qa_rows]
    overview_latencies = [float(row["latency_ms"]) for row in overview_rows]
    sql_latencies = [float(row["latency_ms"]) for row in sql_rows]
    planner_latencies = [float(row["latency_ms"]) for row in planner_rows]
    all_latencies = qa_latencies + overview_latencies + sql_latencies
    all_runtime_rows = qa_rows + overview_rows + sql_rows + [row for row in hard_rows if row.get("latency_ms") is not None]

    answer_accuracy = mean([float(row["answer_key_match"]) for row in qa_rows])
    source_recall = mean([float(row["source_recall@5"]) for row in qa_rows])
    source_mrr = mean([float(row["source_mrr"]) for row in qa_rows])
    citation_validity = mean([float(row["citation_validity"]) for row in qa_rows + overview_rows])
    citation_coverage = mean([float(row["citation_coverage"]) for row in qa_rows + overview_rows])
    overview_pass = mean([float(row["pass"]) for row in overview_rows])
    abstention_accuracy = mean([float(row["pass"]) for row in abstention_rows])
    sql_accuracy = mean([float(row["answer_key_match"]) for row in sql_rows]) if sql_rows else 1.0

    hard_accuracy = mean([float(row.get("pass", False)) for row in hard_rows]) if hard_rows else 1.0

    deterministic_score = (
        0.19 * answer_accuracy
        + 0.12 * source_recall
        + 0.06 * source_mrr
        + 0.08 * citation_validity
        + 0.07 * citation_coverage
        + 0.11 * planner_metrics["planner_route_accuracy"]
        + 0.07 * planner_metrics["planner_task_accuracy"]
        + 0.07 * planner_metrics["planner_strategy_accuracy"]
        + 0.07 * planner_metrics["web_use_precision"]
        + 0.04 * overview_pass
        + 0.04 * abstention_accuracy
        + 0.03 * sql_accuracy
        + 0.05 * hard_accuracy
    )

    judge_rows = [row for row in qa_rows + overview_rows if "judge_overall" in row]
    judge_summary: dict[str, float] = {}
    if judge_rows:
        judge_summary = {
            "judge_faithfulness": mean([float(row["judge_faithfulness"]) for row in judge_rows]),
            "judge_answer_relevance": mean([float(row["judge_answer_relevance"]) for row in judge_rows]),
            "judge_completeness": mean([float(row["judge_completeness"]) for row in judge_rows]),
            "judge_citation_support": mean([float(row["judge_citation_support"]) for row in judge_rows]),
            "judge_overall": mean([float(row["judge_overall"]) for row in judge_rows]),
            "judge_pass_rate": mean([float(row["judge_pass"]) for row in judge_rows]),
            "judge_latency_mean_ms": mean([float(row.get("judge_latency_ms", 0.0)) for row in judge_rows]),
        }

    pacing_stats = request_pacer.stats()

    metrics_for_gate = {
        "planner_route_accuracy": planner_metrics["planner_route_accuracy"],
        "web_use_precision": planner_metrics["web_use_precision"],
        "planner_task_accuracy": planner_metrics["planner_task_accuracy"],
        "citation_validity": citation_validity,
        "citation_coverage": citation_coverage,
        "text2sql_pass_rate": sql_accuracy,
        "hard_mode_pass_rate": hard_accuracy,
    }
    grade, quality_gates = _grade_with_gates(deterministic_score, metrics_for_gate)

    adaptive_budget_row = next(
        (row for row in context_budget_rows if row.get("configuration") == "Adaptive budget"), {}
    )
    compression_row = next(
        (row for row in compression_rows if row.get("configuration") == "Adaptive + sentence compression"), {}
    )
    largest_scale_row = scale_stress_rows[-1] if scale_stress_rows else {}

    summary: dict[str, Any] = {
        "benchmark_version": benchmark.get("version"),
        "evaluation_level": level,
        "deterministic_quality_score": round(deterministic_score, 3),
        "quality_grade": grade,
        "quality_gate_notes": quality_gates,
        "answer_accuracy": round(answer_accuracy, 3),
        "source_precision@5": round(mean([float(row["source_precision@5"]) for row in qa_rows]), 3),
        "source_recall@5": round(source_recall, 3),
        "source_hit@1": round(mean([float(row["source_hit@1"]) for row in qa_rows]), 3),
        "source_mrr": round(source_mrr, 3),
        "source_ap@5": round(mean([float(row["source_ap@5"]) for row in qa_rows]), 3),
        "source_ndcg@5": round(mean([float(row["source_ndcg@5"]) for row in qa_rows]), 3),
        "source_duplicate_rate@5": round(mean([float(row["source_duplicate_rate@5"]) for row in qa_rows]), 3),
        "context_pruning_precision@5": round(float(adaptive_budget_row.get("source_precision@5", 0.0)), 3),
        "context_pruning_recall@5": round(float(adaptive_budget_row.get("source_recall@5", 0.0)), 3),
        "context_pruning_token_reduction_pct": round(float(adaptive_budget_row.get("median_context_reduction_pct", 0.0)), 1),
        "adaptive_context_target_p50": round(float(adaptive_budget_row.get("median_target_chunks", 0.0)), 1),
        "compression_signal_retention": round(float(compression_row.get("answer_signal_retention", 0.0)), 3),
        "compression_additional_reduction_pct": round(float(compression_row.get("median_additional_reduction_pct", 0.0)), 1),
        "scale_stress_max_chunks": int(largest_scale_row.get("chunks", 0) or 0),
        "scale_stress_recall@5": round(float(largest_scale_row.get("source_recall@5", 0.0)), 3) if largest_scale_row else 0.0,
        "scale_stress_pruned_recall@5": round(float(largest_scale_row.get("adaptive_pruned_recall@5", 0.0)), 3) if largest_scale_row else 0.0,
        "citation_validity": round(citation_validity, 3),
        "citation_coverage": round(citation_coverage, 3),
        **{key: round(value, 3) for key, value in planner_metrics.items()},
        "overview_pass_rate": round(overview_pass, 3),
        "abstention_accuracy": round(abstention_accuracy, 3),
        "text2sql_pass_rate": round(sql_accuracy, 3),
        "hard_mode_pass_rate": round(hard_accuracy, 3),
        "profile_benchmark_enabled": bool(profile_rows),
        "profile_benchmark_cases": len(profile_rows),
        "latency_p50_ms": round(percentile(all_latencies, 0.50), 1),
        "latency_p95_ms": round(percentile(all_latencies, 0.95), 1),
        "planner_latency_p50_ms": round(percentile(planner_latencies, 0.50), 1),
        "planner_latency_p95_ms": round(percentile(planner_latencies, 0.95), 1),
        "mean_llm_calls_estimate": round(
            mean([float(row.get("llm_calls_estimate", 0)) for row in all_runtime_rows]), 2
        ),
        "focused_context_tokens_before_p50": round(percentile([float(row.get("context_tokens_est_before", 0)) for row in qa_rows], 0.50), 1),
        "focused_context_tokens_after_p50": round(percentile([float(row.get("context_tokens_est_after", 0)) for row in qa_rows], 0.50), 1),
        "focused_context_pruning_rate": round(mean([float(bool(row.get("context_pruning_used", False))) for row in qa_rows]), 3),
        "focused_generation_prompt_tokens_p50": round(percentile([float(row.get("generation_prompt_tokens_est", 0)) for row in qa_rows], 0.50), 1),
        "focused_generation_total_tokens_p50": round(percentile([float(row.get("generation_total_tokens_est", 0)) for row in qa_rows], 0.50), 1),
        "focused_evidence_utilization_p50": round(percentile([float(row.get("evidence_source_utilization_rate", 0.0)) for row in qa_rows], 0.50), 3),
        "focused_evidence_compression_rate": round(mean([float(bool(row.get("evidence_compression_used", False))) for row in qa_rows]), 3),
        "focused_evidence_compression_reduction_p50": round(percentile([float(row.get("evidence_compression_reduction_pct", 0.0)) for row in qa_rows], 0.50), 1),
        "correction_rate": round(mean([float(row.get("correction_used", False)) for row in all_runtime_rows]), 3),
        "runtime_web_use_rate": round(mean([float(row.get("web_used", False)) for row in all_runtime_rows]), 3),
        "cache_bypassed": True,
        "evaluation_wall_ms": 0.0,
        "evaluation_target_rpm": int(pacing_stats["target_rpm"]),
        "gemini_requests": int(pacing_stats["gemini_requests"]),
        "pacing_sleep_ms": float(pacing_stats["pacing_sleep_ms"]),
        "rate_limit_retries": int(pacing_stats["rate_limit_retries"]),
        "rate_limit_sleep_ms": float(pacing_stats["rate_limit_sleep_ms"]),
        "deep_judge_cases": len(judge_rows),
        "scale_stress_error": scale_stress_error,
        **{key: round(value, 3) for key, value in judge_summary.items()},
    }

    node_latency_rows = _node_latency_summary(qa_rows + overview_rows + hard_rows)
    profile_summary_rows = _profile_summary(profile_rows)
    profile_recommendation = _profile_recommendation(profile_summary_rows)
    if profile_recommendation:
        summary["profile_recommendation"] = profile_recommendation
    if level == "Quick":
        readiness_rows = []
        summary["release_readiness"] = "NOT RUN"
        summary["release_readiness_score"] = None
    else:
        readiness_rows = _readiness_rows(summary, scale_stress_rows, compression_rows)
        readiness_status, readiness_score = _readiness_summary(readiness_rows)
        summary["release_readiness"] = readiness_status
        summary["release_readiness_score"] = round(readiness_score, 3)
    diagnostics = _diagnostics(
        summary, ablation_rows, planner_rows, sql_rows, hard_rows, profile_summary_rows, node_latency_rows,
        context_budget_rows, compression_rows, scale_stress_rows
    )
    summary["evaluation_wall_ms"] = round((time.perf_counter() - wall_started) * 1000, 1)
    progress(1.0, "Evaluation complete")
    return {
        "summary": summary,
        "diagnostics": diagnostics,
        "focused_qa": qa_rows,
        "semantic_planner": planner_rows,
        "corpus_overviews": overview_rows,
        "text2sql": sql_rows,
        "abstention": abstention_rows,
        "retrieval_ablation": ablation_rows,
        "context_budget_ablation": context_budget_rows,
        "evidence_compression_ablation": compression_rows,
        "scale_stress": scale_stress_rows,
        "release_readiness": readiness_rows,
        "hard_mode": hard_rows,
        "profile_benchmark": profile_rows,
        "profile_summary": profile_summary_rows,
        "node_latency": node_latency_rows,
        "methodology": {
            "deterministic": (
                "Transparent labels for answer terms, relevant source files, route/task/strategy, web-use policy, "
                "citation validity/coverage, abstention and latency. Source metrics deduplicate repeated chunks from "
                "the same file before source-level AP/MRR/nDCG are computed."
            ),
            "latency": (
                "Evaluation bypasses the response cache and does not mutate chat history, so reported pipeline latency "
                "reflects real benchmark execution rather than cached answers. Node-latency summaries subtract deliberate "
                "quota pacing proportionally across nodes that issued model calls; raw traces retain wall-clock node time."
            ),
            "deep_judge": (
                "Optional Gemini judge for a representative labeled subset of benchmark cases, covering focused QA, "
                "NIST, cross-document synthesis and corpus overview. Sampling reduces free-tier request pressure while "
                "citation-support and overall scores remain calibrated against deterministic citation validity/coverage. "
                "When a compatible Standard report is supplied, Deep reuses that deterministic baseline and only runs "
                "the sampled judge layer."
            ),
            "quota_safety": (
                f"All Gemini calls in this run share a rolling request pacer targeting {int(pacing_stats['target_rpm'])} RPM. "
                "The pacer also accounts for recent interactive requests recorded by this process and surfaced 429s "
                "honor provider retry guidance before a bounded retry."
            ),
            "text2sql": (
                "Text2SQL routing is evaluated in the semantic-planner suite. The Text2SQL component suite uses one "
                "model call per case to generate validated read-only SQL, executes it in DuckDB and checks labeled scalar "
                "outputs as typed boolean/numeric/text values when available."
            ),
            "evaluation_cache": (
                "Completed reports can be saved by the workspace with corpus/model/benchmark metadata. Saved evaluation "
                "history is separate from the RAG response cache and can be reused without rerunning the benchmark."
            ),
            "quality_gates": (
                "The letter grade is capped when a critical subsystem is weak, preventing a high weighted average "
                "from hiding poor Text2SQL, routing or citation performance."
            ),
            "hard_mode": "Hard-mode cases cover paraphrase, distractors, missing answers, multi-hop comparison, analytical synthesis, structured filtering, local freshness semantics and prompt-injection detection.",
            "profile_benchmark": "Optional Fast/Balanced/Agentic comparison uses a small labeled subset because it intentionally spends additional Gemini requests.",
            "chunk_ablation": "Retrieval ablation reports source-level metrics plus chunk Hit@1/MRR for cases with explicit chunk-content labels.",
            "context_budget": "Focused-query pruning is evaluated as a zero-Gemini ablation across full top-k, a fixed 3-chunk budget, and the adaptive budget chosen from retrieval confidence, score separation and corpus scale.",
            "evidence_compression": "Focused evidence compression selects query-relevant sentences after context budgeting and is evaluated by deterministic answer-signal retention plus token reduction; it never spends a Gemini request.",
            "scale_stress": "Scale stress reuses existing embedding vectors and clones long-document distractor chunks to exercise the real Qdrant + BM25 path at roughly 1x, 5x and 20x distractor scale without additional Gemini calls.",
            "release_readiness": "A transparent readiness checklist applies explicit thresholds to answer quality, grounding, routing, robustness, adaptive-budget recall, compression retention and largest-scale retrieval recall. It does not replace the underlying metrics.",
            "benchmark_file": "evals/demo_benchmark.json",
        },
    }
