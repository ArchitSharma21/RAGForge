import json
from pathlib import Path


def test_v16_benchmark_has_hard_mode_and_insight_plans():
    data = json.loads(Path("evals/demo_benchmark.json").read_text(encoding="utf-8"))
    assert data["version"] == "2.0.3"
    assert len(data.get("hard_mode_cases", [])) >= 8
    insight = [c for c in data["planner_cases"] if c.get("task") == "insight_synthesis"]
    assert insight
    assert all(c.get("strategy") == "analytical" for c in insight)
    assert any(c.get("kind") == "security" for c in data["hard_mode_cases"])
    assert any(c.get("kind") == "missing" for c in data["hard_mode_cases"])


def test_v16_chunk_level_labels_exist_for_reranker_ablation():
    data = json.loads(Path("evals/demo_benchmark.json").read_text(encoding="utf-8"))
    labeled = [c for c in data["qa_cases"] if c.get("chunk_must_contain")]
    assert len(labeled) >= 4


def test_v16_task_schema_includes_analytical_synthesis():
    text = Path("src/ragforge/schemas.py").read_text(encoding="utf-8")
    assert '"insight_synthesis"' in text
    assert '"analytical"' in text


def test_v16_table_citations_are_supported_everywhere():
    from ragforge.citations import normalize_citation_syntax
    from ragforge.eval_metrics import extract_citation_ids, citation_metrics

    assert normalize_citation_syntax("Evidence [D1, T1]. [T1]") == "Evidence [D1] [T1]."
    assert extract_citation_ids("Value is 199 [T1].") == ["T1"]
    metrics = citation_metrics(
        "The Business tier costs 199 [T1].",
        [{"id": "T1", "type": "table", "title": "support_matrix"}],
    )
    assert metrics["citation_validity"] == 1.0
    assert metrics["citation_coverage"] == 1.0


def test_v16_sql_workspace_exposes_deterministic_analytics_context():
    import pytest
    pytest.importorskip("duckdb")
    import pandas as pd
    from ragforge.sql_agent import SQLWorkspace

    ws = SQLWorkspace()
    ws.add_dataframe("support", pd.DataFrame({"tier": ["A", "B"], "price": [10, 20], "weekend": [False, True]}))
    context, sources = ws.analytics_context()
    assert "[T1] TABLE: support" in context
    assert "price: min=10" in context
    assert sources[0]["id"] == "T1"
    assert sources[0]["type"] == "table"


def test_v16_ui_exposes_hard_profile_latency_and_history_tabs():
    text = Path("src/ragforge/ui.py").read_text(encoding="utf-8")
    for label in ["Hard mode", "Profile benchmark", "Node latency", "Evaluation history"]:
        assert f'gr.Tab("{label}")' in text
    assert "Also compare Fast / Balanced / Agentic profiles" in text
    assert "Node latency waterfall" in text


def test_v16_api_exposes_evaluation_history_and_profile_option():
    api = Path("src/ragforge/api.py").read_text(encoding="utf-8")
    schemas = Path("src/ragforge/schemas.py").read_text(encoding="utf-8")
    assert "/api/v1/evaluation/history/{session_id}" in api
    assert "include_profile_benchmark" in schemas
    assert "include_profile_benchmark=payload.include_profile_benchmark" in api
