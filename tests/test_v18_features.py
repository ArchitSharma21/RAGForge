from pathlib import Path

from ragforge.context_budget import focused_context_budget
from ragforge.eval_metrics import missing_answer_match
from ragforge.schemas import Chunk, PipelineConfig, QueryPlan, SearchHit

ROOT = Path(__file__).resolve().parents[1]


def _hit(idx: int, source: str) -> SearchHit:
    return SearchHit(
        chunk=Chunk(id=f"c{idx}", text=("evidence " * (idx + 3)).strip(), source=source),
        score=1.0 / (idx + 1),
        dense_score=max(0.1, 0.9 - idx * 0.1),
        sparse_score=max(0.0, 0.8 - idx * 0.08),
    )


def _plan(task="fact_lookup", strategy="semantic") -> QueryPlan:
    return QueryPlan(
        route="documents",
        knowledge_scope="corpus",
        task_type=task,
        retrieval_strategy=strategy,
        web_relevance="irrelevant",
        rewritten_query="test",
        document_queries=["test"],
    )


def test_v18_focused_context_budget_prunes_tail_but_keeps_safety_floor():
    hits = [_hit(i, "a.md" if i == 0 else f"d{i}.md") for i in range(6)]
    decision = focused_context_budget(hits, _plan(), PipelineConfig(top_k=6, use_context_pruning=True))
    assert decision.used is True
    assert decision.chunks_before == 6
    assert decision.chunks_after == 3
    assert decision.hits == hits[:3]
    assert decision.tokens_est_after < decision.tokens_est_before
    assert decision.reduction_ratio > 0


def test_v18_context_budget_does_not_prune_broad_tasks():
    hits = [_hit(i, f"d{i}.md") for i in range(6)]
    decision = focused_context_budget(
        hits,
        _plan(task="insight_synthesis", strategy="analytical"),
        PipelineConfig(top_k=6, use_context_pruning=True),
    )
    assert decision.used is False
    assert decision.hits == hits
    assert decision.reason == "broad_or_multi_source_task"


def test_v18_context_budget_can_be_disabled():
    hits = [_hit(i, f"d{i}.md") for i in range(6)]
    decision = focused_context_budget(hits, _plan(), PipelineConfig(top_k=6, use_context_pruning=False))
    assert decision.used is False
    assert decision.reason == "disabled_by_user"


def test_v18_exact_grounded_absence_phrase_from_user_report_matches():
    answer = (
        "Based on the provided evidence, the retrieved documents do not mention any fee charged by OrbitPay "
        "for opening a card dispute [D1]. Therefore, the supplied context is insufficient to answer the question."
    )
    assert missing_answer_match(answer, {"expected_missing_any": ["not specified"]})


def test_v18_evaluation_contains_context_budget_ablation_and_grounded_absence_fallback():
    text = (ROOT / "src/ragforge/evaluation.py").read_text(encoding="utf-8")
    assert "_context_budget_ablation" in text
    assert 'result.trace.get("metrics", {}).get("grounded_absence", False)' in text
    assert '"context_budget_ablation": context_budget_rows' in text


def test_v18_ui_exposes_context_budget_controls_and_table():
    text = (ROOT / "src/ragforge/ui.py").read_text(encoding="utf-8")
    assert "Focused context pruning (adaptive)" in text
    assert 'with gr.Tab("Context budget")' in text
    assert '"Context budget ablation": "context_budget_ablation"' in text
