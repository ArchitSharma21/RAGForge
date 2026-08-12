import pytest

pytest.importorskip("langgraph")

from ragforge.pipeline import RAGEngine
from ragforge.schemas import PipelineConfig


class _SQLStub:
    tables = {}


class _EmptyWorkspace:
    chunks = []
    source_profiles = {}
    sql = _SQLStub()
    history = []

    def manifest(self, *args, **kwargs):
        return "No unstructured documents are indexed. Structured tables: none."


def test_document_route_preflight_abstains_before_retrieval():
    engine = object.__new__(RAGEngine)
    engine.workspace = _EmptyWorkspace()
    state = {
        "query": "What are the documents about?",
        "config": PipelineConfig(mode="Documents", profile="Fast"),
        "trace": {"nodes": []},
    }
    out = engine._route(state)
    assert out["route"] == "documents"
    assert out["abstain_reason"] == "workspace_empty_documents"


def test_abstain_is_terminal_operational_response():
    engine = object.__new__(RAGEngine)
    engine.workspace = _EmptyWorkspace()
    state = {
        "abstain_reason": "workspace_empty_documents",
        "doc_hits": [],
        "web_hits": [],
        "trace": {"nodes": []},
    }
    out = engine._abstain(state)
    assert "No document corpus is indexed" in out["answer"]
    assert out["sources"] == []
    assert out["confidence"] >= 0.9
    assert out["trace"]["nodes"][-1]["node"] == "abstain"
