from pathlib import Path

from ragforge.context_budget import adaptive_context_budget, adaptive_retrieval_top_k
from ragforge.evidence_compression import compress_text_for_query, focused_evidence_compression
from ragforge.schemas import Chunk, PipelineConfig, QueryPlan, SearchHit

ROOT = Path(__file__).resolve().parents[1]


def _plan(task="fact_lookup", strategy="semantic") -> QueryPlan:
    return QueryPlan(
        route="documents",
        knowledge_scope="corpus",
        task_type=task,
        retrieval_strategy=strategy,
        web_relevance="irrelevant",
        rewritten_query="Sev-1 acknowledgement target",
        document_queries=["Sev-1 acknowledgement target"],
    )


def _hit(idx: int, source: str, dense: float, sparse: float, text: str | None = None) -> SearchHit:
    return SearchHit(
        chunk=Chunk(
            id=f"c{idx}",
            source=source,
            text=text or ("generic evidence about service operations. " * 35),
        ),
        score=max(dense, sparse),
        dense_score=dense,
        sparse_score=sparse,
    )


def test_v19_adaptive_budget_can_use_two_chunks_for_clear_small_corpus_match():
    hits = [
        _hit(0, "acme.md", 0.95, 0.95),
        _hit(1, "acme.md", 0.55, 0.30),
        _hit(2, "nist.pdf", 0.40, 0.10),
        _hit(3, "other.md", 0.30, 0.05),
    ]
    decision = adaptive_context_budget(
        hits,
        _plan(),
        PipelineConfig(top_k=6, use_context_pruning=True),
        corpus_chunks=90,
        corpus_sources=5,
    )
    assert decision.used is True
    assert decision.target_chunks == 2
    assert decision.hits == hits[:2]
    assert decision.corpus_scale == "small"
    assert decision.retrieval_confidence > 0.75


def test_v19_adaptive_budget_expands_safety_floor_for_large_ambiguous_corpus():
    hits = [
        _hit(0, "a.md", 0.62, 0.45),
        _hit(1, "b.md", 0.61, 0.43),
        _hit(2, "c.md", 0.59, 0.40),
        _hit(3, "d.md", 0.55, 0.35),
        _hit(4, "e.md", 0.50, 0.30),
        _hit(5, "f.md", 0.45, 0.25),
    ]
    decision = adaptive_context_budget(
        hits,
        _plan(),
        PipelineConfig(top_k=6, use_context_pruning=True),
        corpus_chunks=2200,
        corpus_sources=45,
    )
    assert decision.target_chunks == 5
    assert decision.chunks_after == 5
    assert decision.corpus_scale == "large"


def test_v19_adaptive_retrieval_depth_scales_beyond_small_corpus_baseline():
    cfg = PipelineConfig(top_k=6, use_adaptive_top_k=True)
    assert adaptive_retrieval_top_k(cfg, _plan(), corpus_chunks=90, corpus_sources=5) == 6
    assert adaptive_retrieval_top_k(cfg, _plan(), corpus_chunks=500, corpus_sources=15) == 8
    assert adaptive_retrieval_top_k(cfg, _plan(), corpus_chunks=1600, corpus_sources=40) == 10
    assert adaptive_retrieval_top_k(cfg, _plan(), corpus_chunks=3500, corpus_sources=90) == 12


def test_v19_sentence_compression_keeps_query_relevant_fact():
    text = (
        "The Checkout API has a monthly availability SLO of 99.95%. "
        "A Sev-1 incident is a complete outage or payment failure. "
        "The on-call engineer must acknowledge a Sev-1 page within 5 minutes. "
        + "Unrelated deployment documentation is included here. " * 40
    )
    compressed = compress_text_for_query("What is the Sev-1 acknowledgement target?", text, max_chars=500)
    assert "5 minutes" in compressed
    assert len(compressed) < len(text)


def test_v19_focused_compression_does_not_touch_broad_synthesis():
    hits = [_hit(0, "a.md", 0.9, 0.8)]
    result = focused_evidence_compression(
        hits,
        _plan(task="insight_synthesis", strategy="analytical"),
        query="What trends stand out?",
        enabled=True,
    )
    assert result.used is False
    assert result.reason == "broad_or_multi_source_task"


def test_v19_evaluation_contains_scale_stress_compression_and_readiness():
    text = (ROOT / "src/ragforge/evaluation.py").read_text(encoding="utf-8")
    assert "scale_stress_retrieval_eval" in text
    assert '"evidence_compression_ablation": compression_rows' in text
    assert '"release_readiness": readiness_rows' in text
    assert '"scale_stress": scale_stress_rows' in text
    assert "Adaptive budget" in text


def test_v19_api_and_ui_surface_operational_diagnostics_and_new_eval_tabs():
    api = (ROOT / "src/ragforge/api.py").read_text(encoding="utf-8")
    ui = (ROOT / "src/ragforge/ui.py").read_text(encoding="utf-8")
    assert "/api/v1/session/{session_id}/diagnostics" in api
    assert 'with gr.Tab("Evidence compression")' in ui
    assert 'with gr.Tab("Scale stress")' in ui
    assert 'with gr.Tab("Acceptance checks")' in ui
    assert "Adaptive retrieval depth" in ui
    assert "Focused evidence sentence compression" in ui


def test_v19_planner_keeps_overview_and_insight_strategies_structurally_consistent():
    text = (ROOT / "src/ragforge/llm.py").read_text(encoding="utf-8")
    assert 'if plan.task_type == "overview"' in text
    assert 'plan.retrieval_strategy = "global"' in text
    assert 'elif plan.task_type == "insight_synthesis"' in text
    assert 'plan.retrieval_strategy = "analytical"' in text


def test_v203_sentence_compression_keeps_sev1_value_and_excludes_inherited_sev2_target():
    text = (
        "A Sev-1 incident is a complete outage, confirmed data-loss event, or payment-processing failure. "
        "The on-call engineer must acknowledge a Sev-1 page within 5 minutes and establish an incident channel within 10 minutes. "
        "A Sev-2 incident is a major degradation affecting at least 5% of requests. "
        "The acknowledgement target is 15 minutes. "
        + "Unrelated recovery documentation. " * 40
    )
    compressed = compress_text_for_query("What is the Sev-1 acknowledgement target?", text, max_chars=500)
    assert "within 5 minutes" in compressed
    assert "acknowledgement target is 15 minutes" not in compressed.lower()
