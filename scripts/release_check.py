#!/usr/bin/env python3
"""Dependency-free consistency checks for the RAGForge release."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_VERSION = "2.0.3"
BENCHMARK_VERSION = "2.0.3"


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"RELEASE CHECK FAILED: {message}")


def main() -> None:
    pyproject = read("pyproject.toml")
    init_py = read("src/ragforge/__init__.py")
    api_py = read("src/ragforge/api.py")
    ui_py = read("src/ragforge/ui.py")
    readme = read("README.md")
    metrics_py = read("src/ragforge/eval_metrics.py")
    benchmark = json.loads(read("evals/demo_benchmark.json"))

    require(f'version = "{APP_VERSION}"' in pyproject, "pyproject version mismatch")
    require(f'__version__ = "{APP_VERSION}"' in init_py, "package version mismatch")
    require(f'version="{APP_VERSION}"' in api_py, "FastAPI version mismatch")
    require(f'`v{APP_VERSION}`' in ui_py, "runtime UI version mismatch")
    require(benchmark.get("version") == BENCHMARK_VERSION, "benchmark version mismatch")

    require("—" not in ui_py and "–" not in ui_py, "UI contains long dash glyphs")
    require("Final portfolio release" not in ui_py, "release-marketing copy remains in UI")
    require("Pre-final readiness" not in ui_py, "readiness billboard remains in UI")
    require("Verified v1.9 grade" not in ui_py, "historical grade billboard remains in UI")
    require("[D#] [T#] [W#] citations" not in ui_py, "internal citation syntax remains in hero copy")
    require("The results describe this demo benchmark only; they are not general accuracy claims." in ui_py, "evaluation scope note missing")
    require("final portfolio release" not in readme.lower(), "README still reads like portfolio-release marketing")
    require("_contains_expected_term" in metrics_py, "boundary-aware answer matcher missing")

    required_docs = [
        "CHANGELOG.md",
        "SECURITY.md",
        "docs/FINAL_RESULTS.md",
        "docs/PORTFOLIO_GUIDE.md",
        "docs/EVALUATION.md",
        "docs/ARCHITECTURE_API.md",
    ]
    for path in required_docs:
        require((ROOT / path).is_file(), f"missing {path}")

    demo_files = [
        "acme_cloud_runbook.md",
        "orbitpay_policy.txt",
        "release_notes.html",
        "support_matrix.csv",
        "NIST_AI_RMF_1.0.pdf",
    ]
    for name in demo_files:
        require((ROOT / "demo_documents" / name).is_file(), f"missing demo file {name}")

    hard_cases = {case.get("id"): case for case in benchmark.get("hard_mode_cases", [])}
    multihop = hard_cases.get("hard_multihop_time_compare", {})
    require(multihop.get("mode") == "Auto", "multi-hop Hard Mode must use Auto")
    require(multihop.get("profile") == "Balanced", "multi-hop Hard Mode must use Balanced")
    require(multihop.get("expected_task") == "comparison", "multi-hop Hard Mode expected task mismatch")
    require(multihop.get("expected_strategy") == "hierarchical", "multi-hop Hard Mode expected strategy mismatch")

    planner_cases = {case.get("id"): case for case in benchmark.get("planner_cases", [])}
    source_lookup = planner_cases.get("plan_find_refund_file", {})
    require(set(source_lookup.get("strategy_any", [])) == {"semantic", "hierarchical"}, "source-localization strategy alternatives missing")

    architecture_fn = re.search(
        r"def _architecture_snapshot\(.*?\n\s*return sid, runtime, curl, runtime_json",
        ui_py,
        flags=re.S,
    )
    require(architecture_fn is not None, "architecture snapshot function missing")
    require(architecture_fn.group(0).count("**RAGForge:**") == 1, "runtime header is duplicated")

    print(f"RAGForge release check PASS - app {APP_VERSION}, benchmark {BENCHMARK_VERSION}.")


if __name__ == "__main__":
    main()
