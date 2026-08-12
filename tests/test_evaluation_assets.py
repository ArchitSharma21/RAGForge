import json
from pathlib import Path


def test_demo_benchmark_is_multilayer_and_auditable():
    path = Path("evals/demo_benchmark.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == "2.0.3"
    assert len(data["qa_cases"]) >= 9
    assert len(data["planner_cases"]) >= 10
    assert len(data["overview_cases"]) >= 2
    assert len(data["sql_cases"]) >= 3
    assert all(case.get("relevant_sources") for case in data["qa_cases"])
    assert all(case.get("route") for case in data["planner_cases"])
    assert all("web_expected" in case for case in data["planner_cases"])
    assert any(len(case["relevant_sources"]) > 1 for case in data["qa_cases"])


def test_evaluation_module_reports_bounded_retrieval_and_quality_gates():
    text = Path("src/ragforge/evaluation.py").read_text(encoding="utf-8")
    required = [
        "source_hit@1",
        "source_recall@5",
        "source_mrr",
        "source_ap@5",
        "source_ndcg@5",
        "source_duplicate_rate@5",
        "citation_validity",
        "citation_coverage",
        "web_use_precision",
        "unnecessary_web_rate",
        "latency_p95_ms",
        "planner_latency_p95_ms",
        "cache_bypassed",
        "quality_gate_notes",
        "diagnostics",
        "judge_citation_support",
    ]
    for metric in required:
        assert metric in text
    assert "use_cache=False" in text
    assert "record_history=False" in text


def test_demo_evaluation_and_introspection_are_available_through_api():
    text = Path("src/ragforge/api.py").read_text(encoding="utf-8")
    assert "/api/v1/evaluate/demo" in text
    assert "/api/v1/evaluation/benchmark" in text
    assert "/api/v1/session/{session_id}" in text
    assert "/api/v1/evaluation/saved/{session_id}" in text
    assert "/api/v1/evaluation/saved/{session_id}/{level}" in text
    assert 'version="2.0.3"' in text


def test_v15_evaluation_cache_and_incremental_deep_are_present():
    eval_text = Path("src/ragforge/evaluation.py").read_text(encoding="utf-8")
    workspace_text = Path("src/ragforge/workspace.py").read_text(encoding="utf-8")
    ui_text = Path("src/ragforge/ui.py").read_text(encoding="utf-8")
    assert "_deep_from_standard_cache" in eval_text
    assert "base_standard_report" in eval_text
    assert "save_evaluation" in workspace_text
    assert "get_evaluation" in workspace_text
    assert "Compare saved runs" in ui_text
    assert "Reuse saved evaluation" in ui_text


def test_evaluation_is_quota_aware_and_reduces_sql_calls():
    eval_text = Path("src/ragforge/evaluation.py").read_text(encoding="utf-8")
    llm_text = Path("src/ragforge/llm.py").read_text(encoding="utf-8")
    sql_text = Path("src/ragforge/sql_agent.py").read_text(encoding="utf-8")
    assert "RequestPacer" in eval_text
    assert "target_rpm" in eval_text
    assert "pacing_sleep_ms" in eval_text
    assert "rate_limit_retries" in eval_text
    assert "deep_judge_cases" in eval_text
    assert "benchmark_query" in sql_text
    assert "one model call per case" in eval_text
    assert "retry in" in llm_text
    assert "_retry_after_seconds" in llm_text
    assert "if self._is_transient(exc):" in llm_text


def test_deep_judge_uses_representative_sample():
    data = json.loads(Path("evals/demo_benchmark.json").read_text(encoding="utf-8"))
    judged_qa = [case for case in data["qa_cases"] if case.get("deep_judge")]
    judged_overviews = [case for case in data["overview_cases"] if case.get("deep_judge")]
    assert 3 <= len(judged_qa) < len(data["qa_cases"])
    assert len(judged_overviews) == 1


def test_v15_text2sql_cases_have_typed_expected_values():
    data = json.loads(Path("evals/demo_benchmark.json").read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in data["sql_cases"]}
    assert cases["sql_fastest_sla"]["expected_scalar"] == "Enterprise"
    assert cases["sql_business_price"]["expected_scalar"] == 199
    assert cases["sql_weekend_support"]["expected_scalar"] is True


def test_v15_pipeline_contains_adaptive_reranking_and_citation_repair():
    text = Path("src/ragforge/pipeline.py").read_text(encoding="utf-8")
    assert "_reranker_decision" in text
    assert "small_corpus_source_and_chunk_benchmark_no_gain" in text
    assert "_repair_missing_citations" in text
    assert "citation_repairs" in text


def test_v151_citation_repair_skips_preamble_colons_and_normalizes_groups():
    pipeline_text = Path("src/ragforge/pipeline.py").read_text(encoding="utf-8")
    citation_text = Path("src/ragforge/citations.py").read_text(encoding="utf-8")
    assert "_normalize_citation_syntax" in pipeline_text
    assert 'stripped.endswith(":")' in citation_text
    assert "group_re" in citation_text


def test_v152_pipeline_imports_citation_helpers_into_runtime_namespace():
    text = Path("src/ragforge/pipeline.py").read_text(encoding="utf-8")
    assert "from .citations import normalize_citation_syntax, repair_missing_citations" in text
    assert "return repair_missing_citations(answer, sources)" in text
