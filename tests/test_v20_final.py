import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_versions_and_release_assets_are_aligned():
    assert 'version = "2.0.3"' in (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '__version__ = "2.0.3"' in (ROOT / "src/ragforge/__init__.py").read_text(encoding="utf-8")
    assert 'version="2.0.3"' in (ROOT / "src/ragforge/api.py").read_text(encoding="utf-8")
    benchmark = json.loads((ROOT / "evals/demo_benchmark.json").read_text(encoding="utf-8"))
    assert benchmark["version"] == "2.0.3"
    assert (ROOT / "CHANGELOG.md").is_file()
    assert (ROOT / "docs/FINAL_RESULTS.md").is_file()
    assert (ROOT / "docs/PORTFOLIO_GUIDE.md").is_file()
    assert (ROOT / "scripts/release_check.py").is_file()


def test_multihop_hard_case_uses_recommended_semantic_path():
    benchmark = json.loads((ROOT / "evals/demo_benchmark.json").read_text(encoding="utf-8"))
    case = next(c for c in benchmark["hard_mode_cases"] if c["id"] == "hard_multihop_time_compare")
    assert case["mode"] == "Auto"
    assert case["profile"] == "Balanced"
    assert case["expected_route"] == "documents"
    assert case["expected_task"] == "comparison"
    assert case["expected_strategy"] == "hierarchical"


def test_source_localization_planner_case_accepts_two_reasonable_strategies():
    benchmark = json.loads((ROOT / "evals/demo_benchmark.json").read_text(encoding="utf-8"))
    case = next(c for c in benchmark["planner_cases"] if c["id"] == "plan_find_refund_file")
    assert set(case["strategy_any"]) == {"semantic", "hierarchical"}


def test_hard_mode_evaluator_honors_case_profile_and_plan_expectations():
    text = (ROOT / "src/ragforge/evaluation.py").read_text(encoding="utf-8")
    assert 'case_mode = str(case.get("mode") or "Documents")' in text
    assert 'case_profile = str(case.get("profile") or "Fast")' in text
    assert 'expected_task = case.get("expected_task")' in text
    assert 'expected_strategy = case.get("expected_strategy")' in text
    assert '"evaluation_profile": cfg.profile' in text


def test_ui_reads_like_a_product_not_a_release_dashboard():
    text = (ROOT / "src/ragforge/ui.py").read_text(encoding="utf-8")
    assert '<div class="hero-title">RAGForge</div>' in text
    assert "Final portfolio release" not in text
    assert "Pre-final readiness" not in text
    assert "Verified v1.9 grade" not in text
    assert "[D#] [T#] [W#] citations" not in text
    assert "Acceptance checks" in text
    assert "The results describe this demo benchmark only; they are not general accuracy claims." in text
    assert 'with gr.Tab("Chat")' in text
    assert "—" not in text
    assert "–" not in text


def test_readme_leads_with_product_and_scoped_evaluation():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert text.count("# RAGForge") == 1
    assert "final portfolio release" not in text.lower()
    assert "Interpreting the benchmark" in text
    assert "not for making claims about general enterprise performance" in text
    assert "15 minutes" in text and "5 minutes" in text
    assert "CHANGELOG.md" in text
