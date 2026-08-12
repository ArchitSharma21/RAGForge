import pytest

pytest.importorskip("langgraph")

from ragforge.pipeline import RAGEngine
from ragforge.schemas import Chunk, QueryPlan, SearchHit


class _WorkspaceStub:
    source_profiles = {"a": object(), "b": object(), "c": object()}


def test_overview_evidence_rewards_source_coverage_not_literal_query_overlap():
    engine = object.__new__(RAGEngine)
    engine.workspace = _WorkspaceStub()
    hits = [
        SearchHit(Chunk("1", "alpha", "a"), score=1.0, dense_score=0.12),
        SearchHit(Chunk("2", "beta", "b"), score=0.9, dense_score=0.10),
        SearchHit(Chunk("3", "gamma", "c"), score=0.8, dense_score=0.11),
    ]
    plan = QueryPlan(task_type="overview", retrieval_strategy="global")
    assessment = engine._assess_evidence(hits, plan, top_k=6)
    assert assessment.source_coverage == 1.0
    assert assessment.sufficient


def test_focused_evidence_does_not_treat_rrf_as_relevance():
    engine = object.__new__(RAGEngine)
    engine.workspace = _WorkspaceStub()
    hits = [SearchHit(Chunk("1", "unrelated", "a"), score=1.0, dense_score=0.05, sparse_score=0.0)]
    plan = QueryPlan(task_type="fact_lookup", retrieval_strategy="semantic")
    assessment = engine._assess_evidence(hits, plan, top_k=6)
    assert not assessment.sufficient
