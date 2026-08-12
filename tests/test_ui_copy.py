from pathlib import Path


def test_ui_avoids_long_dash_glyphs():
    text = Path("src/ragforge/ui.py").read_text(encoding="utf-8")
    assert "—" not in text
    assert "–" not in text


def test_ui_has_visible_indexing_query_and_eval_states():
    text = Path("src/ragforge/ui.py").read_text(encoding="utf-8")
    assert 'gr.Button(value="Building corpus..."' in text
    assert 'gr.Button(value="Processing..."' in text
    assert "Processing request..." in text
    assert "Running {level} evaluation..." in text
    assert "Please keep this tab open" in text
    assert "BrowserState" in text


def test_ui_prevents_duplicate_long_running_clicks():
    text = Path("src/ragforge/ui.py").read_text(encoding="utf-8")
    assert 'interactive=False' in text
    assert 'show_progress="hidden"' in text
    assert "begin_ask" in text
    assert "begin_eval" in text


def test_ui_exposes_layered_evaluation_and_diagnostics():
    text = Path("src/ragforge/ui.py").read_text(encoding="utf-8")
    for label in ["Focused QA", "Semantic planner", "Corpus overview", "Text2SQL", "Retrieval ablation", "Abstention", "Compare saved runs"]:
        assert label in text
    assert '["Quick", "Standard", "Deep"]' in text
    assert "Diagnostic findings" in text
    assert "Diagnostic findings" in text


def test_architecture_api_tab_is_interactive_and_live():
    text = Path("src/ragforge/ui.py").read_text(encoding="utf-8")
    assert "Refresh runtime view" in text
    assert "Pipeline architecture" in text
    assert "Endpoint reference" in text
    assert "Copy-ready curl examples" in text
    assert "Runtime snapshot" in text
    assert "Evaluation architecture" in text


def test_ui_has_quota_safe_evaluation_controls_and_score_card_spacing():
    text = Path("src/ragforge/ui.py").read_text(encoding="utf-8")
    assert "Quota-safe pacing" in text
    assert "Target Gemini requests per minute" in text
    assert "Run an evaluation to see a summary" in text
    assert "Evaluation score card" not in text


def test_architecture_snapshot_returns_complete_runtime_payload():
    text = Path("src/ragforge/ui.py").read_text(encoding="utf-8")
    assert '"ragforge_version": "2.0.3"' in text
    assert "return sid, runtime, curl, runtime_json" in text
    assert "curl = f\ndef _eval_frame" not in text


def test_ui_can_switch_saved_evaluations_without_rerunning():
    text = Path("src/ragforge/ui.py").read_text(encoding="utf-8")
    assert "View saved evaluation" in text
    assert "Refresh saved runs" in text
    assert "0 new Gemini requests were used" in text


def test_ui_exposes_copyable_evaluation_exports():
    text = Path("src/ragforge/ui.py").read_text(encoding="utf-8")
    assert "Copy / export evaluation tables" in text
    assert "Prepare table for copy / download" in text
    assert "Copy-ready table" in text
    assert "Evaluation report JSON" in text
    assert "pretty_json(report)" in text
    for label in [
        "Focused QA",
        "Semantic planner",
        "Corpus overview",
        "Text2SQL",
        "Retrieval ablation",
        "Abstention",
        "Compare saved runs",
    ]:
        assert label in text
