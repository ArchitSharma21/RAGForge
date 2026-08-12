import pytest

pytest.importorskip("langgraph")

from ragforge.pipeline import RAGEngine
from ragforge.schemas import Chunk, SearchHit


def test_source_formatting():
    hit = SearchHit(Chunk("1", "Evidence text", "file.md"), score=0.9)
    context = RAGEngine._format_context([hit], [])
    assert "[D1]" in context
    assert "file.md" in context


def test_evidence_strength_does_not_treat_rrf_rank_as_calibrated_relevance():
    # A top RRF result may have score=1.0 purely because it ranked first.
    weak = SearchHit(Chunk("1", "unrelated", "file.md"), score=1.0, dense_score=0.08, sparse_score=0.0)
    assert RAGEngine._evidence_strength([weak], []) < 0.30


def test_pipeline_citation_repair_helper_is_runtime_bound():
    sources = [
        {
            "id": "D1",
            "title": "policy.txt",
            "snippet": "Refunds usually appear within 5 to 10 business days after confirmation.",
        }
    ]
    repaired, count = RAGEngine._repair_missing_citations(
        "Refunds usually appear within 5 to 10 business days after confirmation.",
        sources,
    )
    assert count >= 1
    assert "[D1]" in repaired
