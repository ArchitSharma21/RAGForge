from pathlib import Path

from ragforge.eval_metrics import citation_metrics, missing_answer_match


def test_v17_markdown_aware_citation_coverage_counts_short_numbered_items():
    answer = (
        "Based on the provided documents, the four core functions are:\n\n"
        "1. **GOVERN** [D1] [D2]\n"
        "2. **MAP** [D1] [D2]\n"
        "3. **MEASURE** [D1] [D2]\n"
        "4. **MANAGE** [D1] [D2]"
    )
    metrics = citation_metrics(answer, [{"id": "D1"}, {"id": "D2"}])
    assert metrics["citation_validity"] == 1.0
    assert metrics["citation_coverage"] == 1.0
    assert metrics["substantive_units"] == 4


def test_v17_citation_coverage_ignores_generic_list_preamble():
    answer = (
        "Based on the demo documents, escalation guidance is contained in:\n\n"
        "* **acme_cloud_runbook.md:** Incident escalation guidance [D1].\n"
        "* **orbitpay_policy.txt:** Security escalation guidance [D2]."
    )
    metrics = citation_metrics(answer, [{"id": "D1"}, {"id": "D2"}])
    assert metrics["citation_coverage"] == 1.0


def test_v17_grounded_missing_answer_matches_natural_absence_language():
    answer = (
        "Based on the provided evidence, the retrieved documents do not mention any fee "
        "charged by OrbitPay for opening a card dispute [D1]. Therefore, the supplied "
        "context is insufficient to answer the question."
    )
    case = {"expected_missing_any": ["not specified", "does not contain"]}
    assert missing_answer_match(answer, case)


def test_v17_ui_uses_plain_source_cards_and_graphical_latency_bars():
    text = Path("src/ragforge/ui.py").read_text(encoding="utf-8")
    assert 'elem_id="source-panel"' in text
    assert "html.escape(_truncate_preview" in text
    assert "latency-track" in text
    assert "latency-fill" in text
    assert "'#' * width" not in text


def test_v17_saved_run_switching_is_user_input_only_and_has_provenance():
    ui = Path("src/ragforge/ui.py").read_text(encoding="utf-8")
    workspace = Path("src/ragforge/workspace.py").read_text(encoding="utf-8")
    assert "eval_saved_level.input(" in ui
    assert "eval_saved_level.change(" not in ui
    assert "Fresh {level} evaluation complete" in ui
    assert "run_id" in workspace
    assert "SERVER_BOOT_ID" in workspace


def test_v17_global_overviews_surface_table_evidence():
    text = Path("src/ragforge/pipeline.py").read_text(encoding="utf-8")
    assert 'plan.task_type == "overview" and self.workspace.sql.tables' in text
    assert "analytics_context(max_rows=12)" in text


def test_v17_grounded_absence_skips_revise_path():
    text = Path("src/ragforge/pipeline.py").read_text(encoding="utf-8")
    assert "grounded_absence" in text
    assert "and not s.get(\"grounded_absence\", False)" in text


def test_v17_diagnostics_put_next_on_following_line():
    text = Path("src/ragforge/ui.py").read_text(encoding="utf-8")
    assert '  **Next:**' in text


def test_v17_citation_coverage_attaches_citation_after_sentence_punctuation():
    answer = "The corpus does not provide complete financial audits. [D1]"
    metrics = citation_metrics(answer, [{"id": "D1"}])
    assert metrics["citation_coverage"] == 1.0


def test_v17_citation_coverage_does_not_split_vs_abbreviation():
    answer = "Operational strictness vs. flexibility is a notable contrast [D1]."
    metrics = citation_metrics(answer, [{"id": "D1"}])
    assert metrics["citation_coverage"] == 1.0
    assert metrics["substantive_units"] == 1
