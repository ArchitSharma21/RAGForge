from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from .config import get_settings
from .evaluation import demo_benchmark_metadata, run_demo_eval
from .json_utils import pretty_json, to_jsonable
from .pipeline import RAGEngine
from .rate_limit import limiter
from .schemas import PipelineConfig
from .workspace import registry

ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = ROOT / "demo_documents"

CSS = """
#hero {max-width: 1220px; margin: 0 auto 14px auto;}
.hero-shell {padding: 20px 22px; border: 1px solid rgba(128,128,128,.22); border-radius: 14px; background: rgba(128,128,128,.025);}
.hero-kicker {font-size: .78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; opacity: .72;}
.hero-title {font-size: 2.3rem; line-height: 1.05; font-weight: 780; margin: 5px 0 8px 0;}
.hero-subtitle {font-size: 1rem; line-height: 1.55; max-width: 900px; opacity: .84;}
.hero-badges {display: flex; flex-wrap: wrap; gap: 7px; margin-top: 13px;}
.hero-badge {font-size: .79rem; padding: 5px 9px; border: 1px solid rgba(128,128,128,.25); border-radius: 999px; background: rgba(128,128,128,.06);}
.baseline-grid {display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 8px; margin: 10px 0 14px 0;}
.baseline-card {padding: 10px 12px; border: 1px solid rgba(128,128,128,.2); border-radius: 11px; background: rgba(128,128,128,.035);}
.baseline-label {font-size: .72rem; text-transform: uppercase; letter-spacing: .05em; opacity: .66;}
.baseline-value {font-size: 1.12rem; font-weight: 720; margin-top: 2px;}
.section-note {font-size: .88rem; opacity: .78; line-height: 1.45;}
.muted {opacity: .75;}
.status-ready {padding: 8px 10px; border-radius: 8px;}
.status-line {padding: 8px 10px; border: 1px solid rgba(128,128,128,.25); border-radius: 8px; margin: 6px 0;}
#source-panel .source-card {padding: 4px 0;}
#source-panel .source-title {font-size: 1rem; font-weight: 650; line-height: 1.35;}
#source-panel .source-meta {font-size: .88rem; opacity: .78; line-height: 1.4; margin-top: 2px;}
#source-panel .source-snippet {font-size: .95rem; line-height: 1.5; margin-top: 8px; white-space: normal;}
.latency-waterfall {display: grid; gap: 7px; margin-top: 8px;}
.latency-row {display: grid; grid-template-columns: minmax(72px, 110px) 1fr minmax(70px, 90px); gap: 8px; align-items: center;}
.latency-label {font-size: .88rem; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;}
.latency-track {height: 9px; border-radius: 999px; background: rgba(128,128,128,.18); overflow: hidden;}
.latency-fill {height: 100%; min-width: 2px; border-radius: 999px; background: var(--primary-500, currentColor);}
.latency-time {font-size: .84rem; text-align: right; opacity: .8;}
.eval-summary {margin: 8px 0 14px 0; padding: 14px 16px; border: 1px solid rgba(128,128,128,.20); border-radius: 12px; background: rgba(128,128,128,.025);}
.eval-summary-title {font-size: 1.08rem; font-weight: 680; margin-bottom: 3px;}
.eval-summary-scope {font-size: .86rem; opacity: .70; line-height: 1.45; margin-bottom: 10px;}
.eval-summary-table {display: grid; grid-template-columns: minmax(140px, 190px) 1fr; row-gap: 7px; column-gap: 14px;}
.eval-summary-row {display: contents;}
.eval-summary-label {font-size: .86rem; font-weight: 620; opacity: .76;}
.eval-summary-value {font-size: .90rem; line-height: 1.4;}
.footer-note {text-align: center; opacity: .68; font-size: .82rem; padding: 15px 0 4px 0;}
@media (max-width: 900px) {.eval-summary-table {grid-template-columns: 1fr;} .eval-summary-row {display: block; margin-bottom: 8px;}}
"""


def _demo_paths() -> list[Path]:
    return sorted([p for p in DEMO_DIR.iterdir() if p.is_file() and p.name != "README.md"])


def _ensure_session(session_id: str | None) -> tuple[str, Any]:
    ws = registry.get(session_id or None)
    return ws.session_id, ws


def _ui_text(text: str) -> str:
    # Keep UI punctuation visually compact even when model/source text contains
    # typographic dash glyphs. The underlying retrieved evidence is unchanged.
    return (text or "").replace("\u2014", " - ").replace("\u2013", " - ")


def _truncate_preview(text: str, limit: int = 420) -> str:
    clean = re.sub(r"\s+", " ", _ui_text(text)).strip()
    if len(clean) <= limit:
        return clean
    cut = clean[: max(1, limit - 3)]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ,.;:") + "..."


def _corpus_markdown(summary, prefix: str | None = None) -> str:
    sources = "\n".join(f"- `{s}`" for s in summary.sources) or "- *(none)*"
    tables = ", ".join(f"`{t}`" for t in summary.tables) or "none"
    status = prefix or "**Corpus ready**"
    return (
        f"{status}\n\n"
        f"**{len(summary.sources)} sources - {summary.documents} document units - {summary.chunks} chunks - "
        f"{summary.source_profiles} source profiles - tables:** {tables}\n\n{sources}"
    )


def _sources_markdown(sources: list[dict[str, Any]]) -> str:
    """Render source cards with plain-text snippets and uniform typography.

    Retrieved Markdown headings such as ``# Acme Cloud`` must never become UI
    headings inside the source panel. All source-controlled text is HTML-escaped
    before rendering, while the surrounding card markup owns the typography.
    """
    if not sources:
        return "*No sources returned.*"
    blocks: list[str] = []
    for source in sources:
        sid = html.escape(str(source.get("id", "?")))
        title = html.escape(_ui_text(str(source.get("title", "Source"))))
        preview = html.escape(_truncate_preview(str(source.get("snippet", ""))))
        source_type = str(source.get("type", "document"))
        page = source.get("page")
        page_text = f" - page {html.escape(str(page))}" if page else ""

        meta: list[str] = []
        if source_type == "web" and source.get("url"):
            url = html.escape(str(source.get("url")), quote=True)
            meta.append(f'<a href="{url}" target="_blank" rel="noopener noreferrer">Open web source</a>')
            meta.append(f"Retrieval rank: #{html.escape(str(source.get('rank', '-')))}")
        elif source_type in {"sql", "table"}:
            meta.append(f"Rows: {html.escape(str(source.get('rows', '-')))}")
            if source.get("schema"):
                meta.append("Schema: " + html.escape(_ui_text(str(source.get("schema")))) )
        else:
            meta.append(f"Retrieval rank: #{html.escape(str(source.get('rank', '-')))}")
            meta.append(f"hybrid signal: {html.escape(str(source.get('retrieval_signal', '-')))}")

        blocks.append(
            '<div class="source-card">'
            f'<div class="source-title">[{sid}] {title}{page_text}</div>'
            f'<div class="source-meta">{" - ".join(meta)}</div>'
            + (f'<div class="source-snippet">{preview}</div>' if preview else '')
            + '</div>'
        )
    return '<hr>'.join(blocks)


def _latency_waterfall(trace: dict[str, Any]) -> str:
    """Render a compact proportional latency bar instead of ASCII hashes."""
    nodes = [n for n in trace.get("nodes", []) if float(n.get("ms", 0.0) or 0.0) >= 0.0]
    if not nodes:
        return "*No node timings available.*"
    max_ms = max(float(n.get("ms", 0.0) or 0.0) for n in nodes) or 1.0
    rows: list[str] = []
    for node in nodes:
        ms = float(node.get("ms", 0.0) or 0.0)
        width = 0.0 if ms <= 0 else max(1.5, min(100.0, 100.0 * ms / max_ms))
        name = html.escape(str(node.get("node", "-")))
        rows.append(
            '<div class="latency-row">'
            f'<div class="latency-label">{name}</div>'
            '<div class="latency-track">'
            f'<div class="latency-fill" style="width:{width:.1f}%"></div>'
            '</div>'
            f'<div class="latency-time">{ms:.0f} ms</div>'
            '</div>'
        )
    return '<div class="latency-waterfall">' + ''.join(rows) + '</div>'


def _inspector_markdown(trace: dict[str, Any]) -> str:
    if not trace:
        return "*Run a query to inspect routing, retrieval and evidence decisions.*"
    workspace = trace.get("workspace", {})
    plan = trace.get("query_plan", {})
    evidence = trace.get("evidence", {})
    nodes = [n.get("node") for n in trace.get("nodes", []) if n.get("node")]
    metrics = trace.get("metrics", {})
    retrieve_node = next((node for node in trace.get("nodes", []) if node.get("node") == "retrieve"), {})
    coverage = float(evidence.get("source_coverage", 0.0) or 0.0)
    table_count = int(workspace.get("tables", 0) or 0)
    return (
        "**Workspace**  \n"
        f"{workspace.get('sources', 0)} sources - {workspace.get('chunks', 0)} chunks - "
        f"{table_count} {'table' if table_count == 1 else 'tables'} - version {workspace.get('version', 0)}\n\n"
        "**Semantic plan**  \n"
        f"Route: `{plan.get('route', '-')}` - scope: `{plan.get('knowledge_scope', '-')}` - "
        f"task: `{plan.get('task_type', '-')}` - strategy: `{plan.get('retrieval_strategy', '-')}` - "
        f"web: `{plan.get('web_relevance', '-')}`\n\n"
        "**Evidence**  \n"
        f"Score: `{float(evidence.get('score', 0.0) or 0.0):.3f}` - source coverage: `{coverage:.0%}` - "
        f"unique sources: `{evidence.get('unique_sources', 0)}`\n\n"
        "**Runtime**  \n"
        f"Node time: `{float(metrics.get('total_node_ms', 0.0) or 0.0):.0f} ms` - "
        f"estimated LLM calls: `{int(metrics.get('llm_calls_estimate', 0) or 0)}` - "
        f"web used: `{bool(metrics.get('web_used', False))}` - "
        f"correction used: `{bool(metrics.get('correction_used', False))}` - "
        f"reranker used: `{bool(metrics.get('reranker_used', False))}` - "
        f"citation repairs: `{int(metrics.get('citation_repairs', 0) or 0)}` - "
        f"grounded absence: `{bool(metrics.get('grounded_absence', False))}`  \n"
        f"Reranker decision: `{retrieve_node.get('reranker_reason', '-')}`  \n"
        f"Context budget: `{retrieve_node.get('context_pruning_reason', '-')}` - "
        f"policy `{retrieve_node.get('context_budget_policy', '-')}` - "
        f"target `{int(retrieve_node.get('context_budget_target_chunks', 0) or 0)}` - "
        f"chunks `{int(retrieve_node.get('context_chunks_before', 0) or 0)} -> {int(retrieve_node.get('context_chunks_after', 0) or 0)}` - "
        f"estimated tokens `{int(retrieve_node.get('context_tokens_est_before', 0) or 0)} -> {int(retrieve_node.get('context_tokens_est_after', 0) or 0)}` - "
        f"reduction `{float(retrieve_node.get('context_reduction_pct', 0.0) or 0.0):.0f}%`  \n"
        f"Retrieval depth: `{int(retrieve_node.get('retrieval_top_k', 0) or 0)}` - corpus scale `{retrieve_node.get('corpus_scale', '-')}` - "
        f"confidence `{float(retrieve_node.get('retrieval_confidence', 0.0) or 0.0):.2f}` - score gap `{float(retrieve_node.get('retrieval_score_gap', 0.0) or 0.0):.2f}`  \n"
        f"Evidence compression: `{retrieve_node.get('evidence_compression_reason', '-')}` - "
        f"tokens `{int(retrieve_node.get('evidence_tokens_est_before_compression', 0) or 0)} -> {int(retrieve_node.get('evidence_tokens_est_after_compression', 0) or 0)}` - "
        f"additional reduction `{float(retrieve_node.get('evidence_compression_reduction_pct', 0.0) or 0.0):.0f}%`\n\n"
        f"**Execution path**  \n`{' -> '.join(nodes) if nodes else '-'}`\n\n"
        "**Node latency waterfall**  \n" + _latency_waterfall(trace)
    )


def _eval_summary_markdown(report: dict[str, Any]) -> str:
    """Render a scoped benchmark summary without marketing-style grades or badges."""
    summary = report.get("summary", {}) if report else {}
    if not summary:
        return '<div class="eval-summary">Run an evaluation to see a summary.</div>'

    level = str(summary.get("evaluation_level", "Evaluation"))
    qa = report.get("focused_qa", []) or []
    planner = report.get("semantic_planner", []) or []
    overviews = report.get("corpus_overviews", []) or []
    sql = report.get("text2sql", []) or []
    hard = report.get("hard_mode", []) or []

    qa_pass = sum(bool(row.get("answer_key_match")) for row in qa)
    route_pass = sum(bool(row.get("route_correct")) for row in planner)
    task_pass = sum(bool(row.get("task_correct")) for row in planner)
    strategy_pass = sum(bool(row.get("strategy_correct")) for row in planner)
    overview_pass = sum(bool(row.get("pass")) for row in overviews)
    sql_pass = sum(bool(row.get("answer_key_match")) for row in sql)
    hard_pass = sum(bool(row.get("pass")) for row in hard)

    rows = [
        ("Focused QA", f"{qa_pass}/{len(qa)} passed" if qa else "not run"),
        ("Planner", (f"route {route_pass}/{len(planner)}, task {task_pass}/{len(planner)}, strategy {strategy_pass}/{len(planner)}" if planner else "not run")),
        ("Corpus overview", f"{overview_pass}/{len(overviews)} passed" if overviews else "not run"),
        ("Text2SQL", f"{sql_pass}/{len(sql)} passed" if sql else "not run"),
        ("Hard mode", f"{hard_pass}/{len(hard)} passed" if hard else "not run"),
        ("Retrieval", f"Recall@5 {float(summary.get('source_recall@5', 0.0)):.0%}; Precision@5 {float(summary.get('source_precision@5', 0.0)):.0%}"),
        ("Service latency", f"p50 {float(summary.get('latency_p50_ms', 0.0)) / 1000:.2f}s; p95 {float(summary.get('latency_p95_ms', 0.0)) / 1000:.2f}s"),
        ("Gemini requests", f"{int(summary.get('gemini_requests', 0) or 0)} at {int(summary.get('evaluation_target_rpm', 0) or 0)} RPM pacing"),
    ]

    if level != "Quick":
        adaptive_target = float(summary.get("adaptive_context_target_p50", 0.0) or 0.0)
        before = float(summary.get("focused_context_tokens_before_p50", 0.0) or 0.0)
        after = float(summary.get("focused_context_tokens_after_p50", 0.0) or 0.0)
        compression_cases = report.get("evidence_compression_ablation", []) or []
        compressed = next((r for r in compression_cases if r.get("configuration") == "Adaptive + sentence compression"), {})
        retention = float(compressed.get("answer_signal_retention", 0.0) or 0.0)
        compression_n = int(compressed.get("cases", 0) or 0)
        scale_chunks = int(summary.get("scale_stress_max_chunks", 0) or 0)
        scale_recall = float(summary.get("scale_stress_recall@5", 0.0) or 0.0)
        rows.extend([
            ("Context policy", f"median target {adaptive_target:.0f} chunks; estimated focused context {before:.0f} -> {after:.0f} tokens"),
            ("Sentence compression", f"labeled answer signal retained in {int(round(retention * compression_n))}/{compression_n} cases" if compression_n else "not run"),
            ("Synthetic scale stress", f"{scale_chunks:,} chunks; Recall@5 {scale_recall:.0%}" if scale_chunks else "not run"),
        ])
    else:
        rows.append(("Extended ablations", "not run in Quick; use Standard for context, compression, and scale-stress checks"))

    body = ''.join(
        f'<div class="eval-summary-row"><div class="eval-summary-label">{html.escape(label)}</div>'
        f'<div class="eval-summary-value">{html.escape(value)}</div></div>'
        for label, value in rows
    )
    scope = (
        "Smoke-test subset of the bundled demo benchmark."
        if level == "Quick"
        else "Bundled demo benchmark; scale stress uses synthetic distractor copies and is not a claim of enterprise-scale accuracy."
    )
    return (
        '<div class="eval-summary">'
        f'<div class="eval-summary-title">{html.escape(level)} evaluation</div>'
        f'<div class="eval-summary-scope">{html.escape(scope)}</div>'
        f'<div class="eval-summary-table">{body}</div>'
        '</div>'
    )

def _eval_diagnostics_markdown(report: dict[str, Any]) -> str:
    diagnostics = report.get("diagnostics", []) if report else []
    if not diagnostics:
        return "*Diagnostics appear after an evaluation run.*"
    lines = ["### Diagnostic findings"]
    for row in diagnostics:
        severity = str(row.get("severity", "info")).upper()
        lines.append(
            f"- **{severity} - {row.get('area', 'benchmark')}:** {row.get('finding', '')}  \n"
            f"  **Next:** {row.get('recommendation', '')}"
        )
    return "\n".join(lines)


def _api_endpoint_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["GET", "/api/health", "Health check", "No"],
            ["GET", "/api/v1/info", "Models, features and service metadata", "No"],
            ["POST", "/api/v1/session", "Create a workspace session", "If configured"],
            ["GET", "/api/v1/session/{session_id}", "Inspect workspace status", "If configured"],
            ["GET", "/api/v1/session/{session_id}/diagnostics", "Inspect corpus scale, index health and capacity", "If configured"],
            ["POST", "/api/v1/ingest", "Upload and index files", "If configured"],
            ["POST", "/api/v1/query", "Run the RAG pipeline", "If configured"],
            ["POST", "/api/v1/evaluate/demo", "Run Quick, Standard or Deep demo evaluation", "If configured"],
            ["GET", "/api/v1/evaluation/benchmark", "Inspect benchmark version and case counts", "No"],
            ["GET", "/api/v1/evaluation/saved/{session_id}", "List saved evaluation runs", "If configured"],
            ["GET", "/api/v1/evaluation/saved/{session_id}/{level}", "Load one saved evaluation report", "If configured"],
            ["GET", "/api/v1/evaluation/history/{session_id}", "Inspect timestamped evaluation history and deltas", "If configured"],
            ["GET", "/docs", "Interactive FastAPI Swagger UI", "No"],
            ["GET", "/openapi.json", "OpenAPI schema", "No"],
            ["GET", "/metrics", "Prometheus metrics", "No"],
        ],
        columns=["Method", "Path", "Purpose", "Bearer auth"],
    )


def _pipeline_stage_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["guard", "Validate query and record prompt-injection signal"],
            ["route", "Semantic planner chooses corpus, SQL, web or hybrid scope"],
            ["plan", "Build independent document and web retrieval queries"],
            ["retrieve", "Global, hierarchical, semantic or analytical local retrieval, adaptive retrieval depth, context budget, evidence compression and reranker policy"],
            ["grade", "Task-aware evidence sufficiency and source-coverage check"],
            ["correct", "Rewrite/re-plan weak local retrieval before web fallback"],
            ["web", "Conditional Ask-the-Web retrieval only when semantically relevant"],
            ["generate", "Grounded Gemini answer with mandatory evidence citations"],
            ["verify", "Confidence and optional Self-RAG faithfulness audit"],
            ["revise", "One bounded evidence-faithful revision"],
            ["abstain", "Terminal no-answer path when local evidence is unavailable/insufficient"],
        ],
        columns=["Node", "Responsibility"],
    )


def _architecture_snapshot(session_id: str | None) -> tuple[str, str, str, dict[str, Any]]:
    sid, ws = _ensure_session(session_id)
    settings = get_settings()
    stats = ws.health_snapshot()
    runtime_json = {
        "ragforge_version": "2.0.3",
        "workspace": stats,
        "models": {
            "generation": settings.default_model,
            "embedding": settings.embedding_model,
            "reranker": settings.reranker_model,
            "native_search": settings.native_search_model,
        },
        "limits": {
            "max_upload_mb": settings.max_upload_mb,
            "max_archive_files": settings.max_archive_files,
            "max_archive_uncompressed_mb": settings.max_archive_uncompressed_mb,
            "session_ttl_minutes": settings.session_ttl_minutes,
        },
        "storage": {
            "runtime_data_dir": str(settings.data_dir),
            "persistent": False,
        },
    }
    runtime = (
        "### Live runtime\n"
        f"**RAGForge:** `v2.0.3` - **workspace:** `{sid[:12]}...` - **status:** `{stats['status']}`\n\n"
        f"**Corpus:** `{stats['sources']}` sources - `{stats['chunks']}` chunks - "
        f"`{stats['source_profiles']}` source profiles - `{stats['tables']}` tables - "
        f"corpus version `{stats['version']}`\n\n"
        f"**Scale / capacity:** `{stats.get('corpus_scale', '-')}` - estimated index memory "
        f"`{float(stats.get('estimated_index_memory_mb', 0.0)):.2f} MB` - chunk capacity "
        f"`{float(stats.get('chunk_capacity_utilization', 0.0)):.0%}` - status `{stats.get('capacity_status', '-')}`\n\n"
        f"**Saved evaluations:** `{', '.join(stats.get('saved_evaluations', [])) or 'none'}`\n\n"
        f"**Models:** generation `{settings.default_model}` - embeddings `{settings.embedding_model}` - "
        f"reranker `{settings.reranker_model}` - native search `{settings.native_search_model}`"
    )
    curl = f"""# Replace with your deployed Space URL
BASE_URL=\"https://YOUR-SPACE.hf.space\"
SESSION_ID=\"{sid}\"

# Health
curl \"$BASE_URL/api/health\"

# Current workspace status
curl \"$BASE_URL/api/v1/session/$SESSION_ID\"

# Query the current workspace
curl -X POST \"$BASE_URL/api/v1/query\" \\
  -H \"Content-Type: application/json\" \\
  -d '{{
    \"session_id\": \"{sid}\",
    \"query\": \"What is the collection about?\",
    \"config\": {{\"mode\": \"Auto\", \"profile\": \"Balanced\"}}
  }}'

# Benchmark metadata
curl \"$BASE_URL/api/v1/evaluation/benchmark\"

# Saved evaluation inventory
curl \"$BASE_URL/api/v1/evaluation/saved/$SESSION_ID\"
"""
    return sid, runtime, curl, runtime_json


def _eval_frame(report: dict[str, Any], key: str) -> pd.DataFrame:
    rows = report.get(key, []) if report else []
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame[[column for column in frame.columns if not str(column).startswith("_")]]
    return frame


EVAL_TABLE_KEYS = {
    "Focused QA": "focused_qa",
    "Semantic planner": "semantic_planner",
    "Corpus overview": "corpus_overviews",
    "Text2SQL": "text2sql",
    "Retrieval ablation": "retrieval_ablation",
    "Context budget ablation": "context_budget_ablation",
    "Evidence compression": "evidence_compression_ablation",
    "Scale stress": "scale_stress",
    "Acceptance checks": "release_readiness",
    "Hard mode": "hard_mode",
    "Profile benchmark": "profile_benchmark",
    "Profile summary": "profile_summary",
    "Node latency": "node_latency",
    "Abstention": "abstention",
    "Compare saved runs": "__compare__",
    "Evaluation history": "__history__",
}


def _eval_table_frame(ws, report: dict[str, Any], label: str) -> pd.DataFrame:
    key = EVAL_TABLE_KEYS.get(label)
    if key == "__compare__":
        return _eval_comparison_frame(ws)
    if key == "__history__":
        return _eval_history_frame(ws)
    if not key:
        return pd.DataFrame()
    return _eval_frame(report, key)


def _table_export_text(frame: pd.DataFrame, export_format: str) -> tuple[str, str]:
    if frame.empty:
        return "", "csv"
    fmt = (export_format or "CSV").upper()
    if fmt == "TSV":
        return frame.to_csv(index=False, sep="\t"), "tsv"
    if fmt == "MARKDOWN":
        return frame.to_markdown(index=False), "md"
    return frame.to_csv(index=False), "csv"


def _eval_comparison_frame(ws) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in ws.evaluation_inventory():
        report = ws.get_evaluation(item["level"], require_current_corpus=False) or {}
        summary = report.get("summary", {})
        rows.append(
            {
                "depth": item["level"],
                "answer_accuracy": summary.get("answer_accuracy"),
                "source_recall@5": summary.get("source_recall@5"),
                "source_precision@5": summary.get("source_precision@5"),
                "planner_strategy_accuracy": summary.get("planner_strategy_accuracy"),
                "hard_mode_pass": summary.get("hard_mode_pass_rate"),
                "text2sql_pass": summary.get("text2sql_pass_rate"),
                "scale_stress_recall@5": summary.get("scale_stress_recall@5"),
                "pipeline_p50_ms": summary.get("latency_p50_ms"),
                "gemini_requests": summary.get("gemini_requests"),
                "pacing_wait_s": round(float(summary.get("pacing_sleep_ms", 0.0) or 0.0) / 1000, 1),
                "deep_judge_overall": summary.get("judge_overall", ""),
                "current_corpus": item.get("current_corpus", False),
                "run_id": item.get("run_id", ""),
                "saved_at": item.get("saved_at", ""),
            }
        )
    return pd.DataFrame(rows)



def _eval_history_frame(ws) -> pd.DataFrame:
    rows = []
    for row in ws.evaluation_history_inventory():
        rows.append({
            "saved_at": row.get("saved_at"),
            "level": row.get("level"),
            "benchmark": row.get("benchmark"),
            "model": row.get("model"),
            "workspace_version": row.get("workspace_version"),
            "citation_coverage": row.get("citation_coverage"),
            "hard_mode_pass": row.get("hard_mode_pass"),
            "p50_ms": row.get("p50_ms"),
            "gemini_requests": row.get("gemini_requests"),
            "run_id": row.get("run_id"),
        })
    return pd.DataFrame(rows)

def _saved_eval_status(ws) -> str:
    inventory = ws.evaluation_inventory()
    if not inventory:
        return "*No saved evaluation runs for this workspace yet.*"
    parts = []
    for item in inventory:
        stale = "" if item.get("current_corpus") else " (stale corpus)"
        run_id = item.get("run_id") or "legacy"
        parts.append(f"`{item['level']}` - run `{run_id}`{stale}")
    return "**Saved runs:** " + " - ".join(parts)


def _saved_eval_message(level: str, report: dict[str, Any], *, stale: bool = False) -> str:
    meta = report.get("evaluation_cache", {}) if report else {}
    run_id = meta.get("run_id") or "legacy"
    saved_at = meta.get("saved_at") or "unknown time"
    suffix = "This result belongs to an older corpus version." if stale else "0 new Gemini requests were used."
    return f"**Loaded saved {level} run `{run_id}` from {saved_at}.** {suffix}"


def build_ui() -> gr.Blocks:
    settings = get_settings()
    with gr.Blocks(css=CSS, title="RAGForge") as demo:
        if hasattr(gr, "BrowserState"):
            session_state = gr.BrowserState("", storage_key="ragforge_session_id_v1_2")
        else:  # Compatibility fallback for older Gradio builds.
            session_state = gr.State("")

        gr.HTML(
            """
<div id="hero" class="hero-shell">
  <div class="hero-title">RAGForge</div>
  <div class="hero-subtitle">
    Search and analyze documents, structured tables, and web sources in one workspace.
    RAGForge routes each question to the appropriate retrieval path and keeps the supporting sources and execution trace visible.
  </div>
  <div class="hero-badges">
    <span class="hero-badge">Hybrid document search</span>
    <span class="hero-badge">Read-only Text2SQL</span>
    <span class="hero-badge">Conditional web search</span>
    <span class="hero-badge">Built-in evaluation</span>
  </div>
</div>
            """
        )

        with gr.Tabs():
            with gr.Tab("Chat"):
                with gr.Row():
                    with gr.Column(scale=4):
                        gr.Markdown("### Corpus")
                        uploads = gr.File(
                            label="Upload documents or a ZIP",
                            file_count="multiple",
                            type="filepath",
                            file_types=[
                                ".pdf", ".txt", ".md", ".rst", ".docx", ".pptx", ".csv", ".xls", ".xlsx",
                                ".json", ".html", ".htm", ".xml", ".yaml", ".yml", ".py", ".js", ".ts",
                                ".java", ".c", ".cpp", ".sql", ".log", ".zip", ".png", ".jpg", ".jpeg", ".webp",
                            ],
                        )
                        use_demo = gr.Checkbox(label="Use bundled demo files", value=True)
                        gr.Markdown(
                            "<small>Includes Acme Cloud runbook, OrbitPay policy, support CSV, release notes, "
                            "and the NIST AI RMF 1.0 PDF. Demo files can auto-initialize on the first question.</small>"
                        )
                        use_ocr = gr.Checkbox(label="Gemini OCR for scanned PDFs/images", value=False)
                        semantic_chunking = gr.Checkbox(label="Semantic breakpoint chunking", value=False)
                        index_btn = gr.Button("Index corpus", variant="primary")
                        reset_btn = gr.Button("Reset session")
                        corpus = gr.Markdown(
                            "No corpus indexed yet. Click **Index corpus**, or leave demo files enabled and ask a question."
                        )

                        gr.Markdown("### Pipeline settings")
                        mode = gr.Dropdown(["Auto", "Documents", "Web", "Hybrid", "Data (SQL)"], value="Auto", label="Route")
                        profile = gr.Radio(["Fast", "Balanced", "Agentic"], value="Balanced", label="Pipeline profile", info="Balanced is the recommended default. Fast minimizes model calls; Agentic enables the richest corrective/verification behavior.")
                        model = gr.Dropdown(
                            ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.6-flash", "gemini-3.5-flash"],
                            value=settings.default_model,
                            label="Gemini model",
                        )
                        web_provider = gr.Dropdown(
                            ["Auto", "DuckDuckGo", "Tavily", "Gemini Search"], value="Auto", label="Web search provider"
                        )
                        api_key = gr.Textbox(
                            label="Gemini API key (optional if Space secret is set)", type="password", placeholder="AIza..."
                        )
                        with gr.Accordion("Advanced RAG switches", open=False):
                            hyde = gr.Checkbox(value=True, label="HyDE")
                            multi_query = gr.Checkbox(value=True, label="Multi-query expansion")
                            reranker = gr.Checkbox(value=True, label="Cross-encoder reranking (adaptive)", info="Fast mode and small/easy corpora may skip the cross-encoder when benchmark evidence shows no ranking gain. Disable this switch to force reranking off entirely.")
                            context_pruning = gr.Checkbox(value=True, label="Focused context pruning (adaptive)", info="For focused local lookups, chooses a 2-5 chunk budget from retrieval confidence, score separation and corpus scale. Overview, insight and comparison tasks keep broad context.")
                            adaptive_top_k = gr.Checkbox(value=True, label="Adaptive retrieval depth", info="Treats Final context chunks as the small-corpus baseline and retrieves a wider candidate set for larger corpora before context budgeting.")
                            evidence_compression = gr.Checkbox(value=True, label="Focused evidence sentence compression", info="After context budgeting, keeps query-relevant sentences for focused local lookups while source cards retain the original chunk text.")
                            crag = gr.Checkbox(value=True, label="CRAG corrective retrieval + conditional web fallback")
                            self_rag = gr.Checkbox(value=True, label="Self-RAG faithfulness check")
                            web_fallback = gr.Checkbox(value=True, label="Allow web fallback")
                            top_k = gr.Slider(2, 12, value=6, step=1, label="Final context chunks")

                    with gr.Column(scale=7):
                        gr.Markdown("### Ask RAGForge")
                        gr.Markdown("<small>Try: `What is the Sev-1 acknowledgement target?` - `What is the collection about?` - `Which support tier has the shortest SLA?`</small>")
                        chatbot = gr.Chatbot(label="Conversation", type="messages", height=510)
                        query = gr.Textbox(label="Question", placeholder="Ask about the indexed corpus, structured tables, or current web information...", lines=2)
                        ask_btn = gr.Button("Ask", variant="primary")
                        query_status = gr.Markdown("Ready.", elem_classes=["status-line"])
                        with gr.Accordion("Sources", open=True):
                            source_view = gr.Markdown("*Sources appear here.*", elem_id="source-panel")
                        with gr.Accordion("Pipeline inspector", open=False):
                            inspector_summary = gr.Markdown(
                                "*Run a query to inspect routing, retrieval and evidence decisions.*"
                            )
                            with gr.Accordion("Raw trace", open=False):
                                inspector = gr.JSON(label="Trace")

                def restore_session(sid):
                    if sid and registry.contains(sid):
                        ws = registry.require(sid)
                        return sid, _corpus_markdown(ws.summary(), "**Session restored - corpus ready**")
                    ws = registry.create()
                    if sid:
                        return (
                            ws.session_id,
                            "**Previous session expired or the Space restarted.**  \n"
                            "Demo mode will rebuild automatically on the next question. Custom uploads must be indexed again.",
                        )
                    return ws.session_id, "No corpus indexed yet. Click **Index corpus**, or leave demo files enabled and ask a question."

                demo.load(restore_session, [session_state], [session_state, corpus])

                def begin_index():
                    return gr.Button(value="Building corpus...", interactive=False)

                def index_files(
                    files,
                    demo_flag,
                    ocr_flag,
                    semantic_flag,
                    sid,
                    key,
                    model_name,
                    request: gr.Request,
                    progress=gr.Progress(),
                ):
                    try:
                        client = getattr(getattr(request, "client", None), "host", None) or "unknown"
                        limiter.check(f"ui-ingest:{client}")
                        sid, ws = _ensure_session(sid)
                        paths = [Path(p) for p in (files or [])]
                        if demo_flag:
                            paths += _demo_paths()
                        if not paths:
                            return (
                                sid,
                                "**Nothing to index.** Upload at least one file or enable the demo corpus.",
                                gr.Button(value="Index corpus", interactive=True),
                            )

                        def report(value: float, message: str) -> None:
                            progress(value, desc=message)

                        summary = ws.ingest(
                            paths,
                            ocr=ocr_flag,
                            semantic_chunking=semantic_flag,
                            api_key=(key or None),
                            model=model_name,
                            progress_callback=report,
                        )
                        return sid, _corpus_markdown(summary), gr.Button(value="Index corpus", interactive=True)
                    except Exception as exc:
                        return (
                            sid or "",
                            f"**Corpus build failed.** `{type(exc).__name__}: {exc}`",
                            gr.Button(value="Index corpus", interactive=True),
                        )

                index_event = index_btn.click(begin_index, None, [index_btn], queue=False, show_progress="hidden")
                index_event.then(
                    index_files,
                    [uploads, use_demo, use_ocr, semantic_chunking, session_state, api_key, model],
                    [session_state, corpus, index_btn],
                    show_progress="hidden",
                )

                def reset_session(sid):
                    if sid and registry.contains(sid):
                        registry.delete(sid)
                    ws = registry.create()
                    return (
                        ws.session_id,
                        [],
                        "No corpus indexed yet. Demo mode can initialize automatically on the next question.",
                        "*Sources appear here.*",
                        "*Run a query to inspect routing, retrieval and evidence decisions.*",
                        {},
                    )

                reset_btn.click(
                    reset_session,
                    [session_state],
                    [session_state, chatbot, corpus, source_view, inspector_summary, inspector],
                )

                def begin_ask(q):
                    if not q or not q.strip():
                        return (
                            gr.Button(value="Ask", interactive=True),
                            gr.Textbox(value=q or "", interactive=True),
                            "**Enter a question first.**",
                        )
                    return (
                        gr.Button(value="Processing...", interactive=False),
                        gr.Textbox(value=q, interactive=False),
                        "**Processing request...** Routing, retrieval and generation are running. This can take a few seconds.",
                    )

                def answer(
                    q,
                    history,
                    sid,
                    demo_flag,
                    ocr_flag,
                    semantic_flag,
                    mode_v,
                    profile_v,
                    model_v,
                    web_v,
                    key,
                    hyde_v,
                    mq_v,
                    rerank_v,
                    context_pruning_v,
                    adaptive_top_k_v,
                    evidence_compression_v,
                    crag_v,
                    selfrag_v,
                    fallback_v,
                    topk_v,
                    request: gr.Request,
                ):
                    sid, ws = _ensure_session(sid)
                    if not q or not q.strip():
                        corpus_md = _corpus_markdown(ws.summary()) if not ws.is_empty else "No corpus indexed yet."
                        return (
                            history, sid, "*No sources returned.*", {}, _inspector_markdown({}), corpus_md,
                            gr.Button(value="Ask", interactive=True),
                            gr.Textbox(value=q or "", interactive=True),
                            "**Enter a question first.**",
                        )
                    try:
                        client = getattr(getattr(request, "client", None), "host", None) or "unknown"
                        limiter.check(client)

                        # Demo mode is self-healing. Browser state can outlive an
                        # ephemeral HF container, so rebuild bundled data lazily.
                        corpus_md = _corpus_markdown(ws.summary()) if not ws.is_empty else "No corpus indexed yet."
                        if ws.is_empty and demo_flag and mode_v != "Web":
                            summary = ws.ingest(
                                _demo_paths(),
                                ocr=ocr_flag,
                                semantic_chunking=semantic_flag,
                                api_key=(key or None),
                                model=model_v,
                            )
                            corpus_md = _corpus_markdown(summary, "**Demo corpus initialized automatically**")

                        cfg = PipelineConfig(
                            mode=mode_v,
                            profile=profile_v,
                            model=model_v,
                            web_provider=web_v,
                            use_hyde=hyde_v,
                            use_multi_query=mq_v,
                            use_reranker=rerank_v,
                            use_context_pruning=context_pruning_v,
                            use_adaptive_top_k=adaptive_top_k_v,
                            use_evidence_compression=evidence_compression_v,
                            use_crag=crag_v,
                            use_self_rag=selfrag_v,
                            allow_web_fallback=fallback_v,
                            top_k=int(topk_v),
                        )
                        result = RAGEngine(ws).ask(q, cfg, api_key=(key or None))
                        hist = list(history or [])
                        hist.append({"role": "user", "content": q})
                        hist.append({"role": "assistant", "content": _ui_text(result.answer)})
                        trace = dict(result.trace)
                        trace["confidence"] = round(result.confidence, 3)
                        status = (
                            f"**Answer ready.** Confidence `{result.confidence:.2f}` - "
                            f"route `{trace.get('query_plan', {}).get('route', '-')}`."
                        )
                        return (
                            hist, sid, _sources_markdown(result.sources), trace, _inspector_markdown(trace), corpus_md,
                            gr.Button(value="Ask", interactive=True),
                            gr.Textbox(value="", interactive=True),
                            status,
                        )
                    except Exception as exc:
                        return (
                            history, sid, "*No sources returned.*", {}, _inspector_markdown({}),
                            _corpus_markdown(ws.summary()) if not ws.is_empty else "No corpus indexed yet.",
                            gr.Button(value="Ask", interactive=True),
                            gr.Textbox(value=q, interactive=True),
                            f"**Request failed.** `{type(exc).__name__}: {exc}`",
                        )

                inputs = [
                    query,
                    chatbot,
                    session_state,
                    use_demo,
                    use_ocr,
                    semantic_chunking,
                    mode,
                    profile,
                    model,
                    web_provider,
                    api_key,
                    hyde,
                    multi_query,
                    reranker,
                    context_pruning,
                    adaptive_top_k,
                    evidence_compression,
                    crag,
                    self_rag,
                    web_fallback,
                    top_k,
                ]
                outputs = [
                    chatbot, session_state, source_view, inspector, inspector_summary, corpus,
                    ask_btn, query, query_status,
                ]
                ask_event = ask_btn.click(
                    begin_ask, [query], [ask_btn, query, query_status], queue=False, show_progress="hidden"
                )
                ask_event.then(answer, inputs, outputs, show_progress="hidden")

                submit_event = query.submit(
                    begin_ask, [query], [ask_btn, query, query_status], queue=False, show_progress="hidden"
                )
                submit_event.then(answer, inputs, outputs, show_progress="hidden")

            with gr.Tab("Evaluation"):
                gr.Markdown(
                    "### Evaluation\n"
                    "Run the bundled benchmark to inspect retrieval, routing, grounding, SQL behavior, robustness, latency, and context policies. "
                    "The results describe this demo benchmark only; they are not general accuracy claims."
                )
                eval_level = gr.Radio(
                    ["Quick", "Standard", "Deep"],
                    value="Standard",
                    label="Evaluation depth",
                    info=(
                        "Quick is a small smoke test. Standard runs the full deterministic benchmark, hard-mode suite and retrieval "
                        "ablation. Deep additionally uses a calibrated Gemini judge."
                    ),
                )
                with gr.Accordion("Evaluation API pacing", open=False):
                    eval_quota_safe = gr.Checkbox(
                        value=True,
                        label="Quota-safe pacing (recommended for free-tier Gemini API keys)",
                    )
                    eval_target_rpm = gr.Slider(
                        4,
                        30,
                        value=12,
                        step=1,
                        label="Target Gemini requests per minute",
                        info=(
                            "Use a value below the active RPM shown for your project in Google AI Studio. "
                            "12 RPM leaves headroom when the active limit is 15 RPM."
                        ),
                    )
                    gr.Markdown(
                        "Quota-safe mode also accounts for recent requests made by this running Space and honors "
                        "Gemini retry guidance when a 429 is returned. Standard and Deep runs can therefore take longer."
                    )
                eval_reuse_saved = gr.Checkbox(
                    value=True,
                    label="Reuse saved evaluation when the corpus, model and benchmark match",
                    info=(
                        "Avoids duplicate Gemini calls. Deep can also reuse a saved Standard deterministic baseline "
                        "and add only the sampled judge layer."
                    ),
                )
                eval_profile_benchmark = gr.Checkbox(
                    value=False,
                    label="Also compare Fast / Balanced / Agentic profiles (extra Gemini calls)",
                    info=(
                        "Runs a small labeled profile benchmark. Keep this off for normal free-tier evaluation; "
                        "enable it when you explicitly want latency/quality/call-count tradeoffs across profiles."
                    ),
                )
                eval_btn = gr.Button("Run evaluation", variant="primary")
                eval_status = gr.Markdown("Ready to evaluate.", elem_classes=["status-line"])
                eval_scorecard = gr.HTML('<div class="eval-summary">Run an evaluation to see a summary.</div>')
                eval_diagnostics = gr.Markdown("*Diagnostics appear after an evaluation run.*")
                with gr.Accordion("Saved evaluation runs", open=True):
                    eval_saved_status = gr.Markdown("*No saved evaluation runs for this workspace yet.*")
                    eval_saved_level = gr.Radio(
                        ["Quick", "Standard", "Deep"],
                        value="Standard",
                        label="View saved evaluation",
                        info="Switch between saved runs without rerunning the benchmark or consuming Gemini quota.",
                    )
                    eval_refresh_saved = gr.Button("Refresh saved runs")
                with gr.Tabs():
                    with gr.Tab("Focused QA"):
                        eval_qa = gr.Dataframe(interactive=False, wrap=True)
                    with gr.Tab("Semantic planner"):
                        eval_planner = gr.Dataframe(interactive=False, wrap=True)
                    with gr.Tab("Corpus overview"):
                        eval_overview = gr.Dataframe(interactive=False, wrap=True)
                    with gr.Tab("Text2SQL"):
                        eval_sql = gr.Dataframe(interactive=False, wrap=True)
                    with gr.Tab("Retrieval ablation"):
                        eval_ablation = gr.Dataframe(interactive=False, wrap=True)
                    with gr.Tab("Context budget"):
                        eval_context_budget = gr.Dataframe(interactive=False, wrap=True)
                    with gr.Tab("Evidence compression"):
                        eval_compression = gr.Dataframe(interactive=False, wrap=True)
                    with gr.Tab("Scale stress"):
                        eval_scale_stress = gr.Dataframe(interactive=False, wrap=True)
                    with gr.Tab("Acceptance checks"):
                        eval_readiness = gr.Dataframe(interactive=False, wrap=True)
                    with gr.Tab("Hard mode"):
                        eval_hard = gr.Dataframe(interactive=False, wrap=True)
                    with gr.Tab("Profile benchmark"):
                        eval_profiles = gr.Dataframe(interactive=False, wrap=True)
                    with gr.Tab("Node latency"):
                        eval_node_latency = gr.Dataframe(interactive=False, wrap=True)
                    with gr.Tab("Abstention"):
                        eval_abstention = gr.Dataframe(interactive=False, wrap=True)
                    with gr.Tab("Compare saved runs"):
                        eval_compare = gr.Dataframe(interactive=False, wrap=True)
                    with gr.Tab("Evaluation history"):
                        eval_history = gr.Dataframe(interactive=False, wrap=True)
                with gr.Accordion("Copy / export evaluation tables", open=False):
                    gr.Markdown(
                        "Choose any evaluation table, prepare it as CSV, TSV or Markdown, then use the copy button "
                        "in the code box or download the file."
                    )
                    with gr.Row():
                        eval_export_table = gr.Dropdown(
                            list(EVAL_TABLE_KEYS),
                            value="Text2SQL",
                            label="Table",
                        )
                        eval_export_format = gr.Radio(
                            ["CSV", "TSV", "Markdown"],
                            value="CSV",
                            label="Format",
                        )
                    eval_export_btn = gr.Button("Prepare table for copy / download")
                    eval_export_status = gr.Markdown("*No table prepared yet.*")
                    eval_export_code = gr.Code(
                        label="Copy-ready table",
                        language=None,
                        interactive=False,
                        lines=12,
                    )
                    eval_export_download = gr.DownloadButton("Download table", value=None)
                with gr.Accordion("Raw evaluation report", open=False):
                    gr.Markdown(
                        "The JSON below is normalized before rendering, so fresh and restored reports use the same "
                        "format. Use the copy button in the code box for a one-click copy."
                    )
                    eval_output = gr.Code(
                        label="Evaluation report JSON",
                        language="json",
                        interactive=False,
                        lines=24,
                    )

                def begin_eval(level, quota_safe, target_rpm, include_profiles):
                    descriptions = {
                        "Quick": "Running Quick evaluation - smoke-testing QA, planner, overview, SQL, hard-mode and abstention.",
                        "Standard": "Running Standard evaluation - full deterministic benchmark, hard-mode robustness, adaptive-context/compression ablations and local scale stress.",
                        "Deep": (
                            "Running Deep evaluation - representative calibrated judge sample. From scratch this is roughly "
                            "the Standard run plus 5 judge calls; with a matching saved Standard baseline it is about 5 judge calls."
                        ),
                    }
                    pacing = (
                        f" Quota-safe pacing is enabled at {int(target_rpm)} RPM."
                        if quota_safe
                        else " Quota-safe pacing is disabled; provider 429s are still retried with backoff."
                    )
                    profile_note = (
                        " Profile comparison is enabled and will spend additional Gemini requests."
                        if include_profiles and level != "Quick"
                        else ""
                    )
                    return (
                        gr.Button(value=f"Running {level} evaluation...", interactive=False),
                        f"**{descriptions.get(level, descriptions['Standard'])}**{pacing}{profile_note} Please keep this tab open.",
                        "*Evaluation is running. Results will replace this message when the run finishes.*",
                    )

                def _evaluation_outputs(ws, report, status_text):
                    report = to_jsonable(report)
                    return (
                        _eval_summary_markdown(report),
                        _eval_diagnostics_markdown(report),
                        _eval_frame(report, "focused_qa"),
                        _eval_frame(report, "semantic_planner"),
                        _eval_frame(report, "corpus_overviews"),
                        _eval_frame(report, "text2sql"),
                        _eval_frame(report, "retrieval_ablation"),
                        _eval_frame(report, "context_budget_ablation"),
                        _eval_frame(report, "evidence_compression_ablation"),
                        _eval_frame(report, "scale_stress"),
                        _eval_frame(report, "release_readiness"),
                        _eval_frame(report, "hard_mode"),
                        _eval_frame(report, "profile_benchmark"),
                        _eval_frame(report, "node_latency"),
                        _eval_frame(report, "abstention"),
                        _eval_comparison_frame(ws),
                        _eval_history_frame(ws),
                        pretty_json(report),
                        _saved_eval_status(ws),
                        status_text,
                    )

                def load_saved_eval(sid, level):
                    sid, ws = _ensure_session(sid)
                    report = ws.get_evaluation(level, require_current_corpus=False)
                    if not report:
                        status = f"**No saved {level} evaluation exists for this workspace.** Run it once to cache it."
                        outputs = _evaluation_outputs(ws, {}, status)
                        return (sid, outputs[-1], *outputs[:-1])
                    meta = report.get("evaluation_cache", {})
                    stale = int(meta.get("workspace_version", -1)) != int(ws.version)
                    status = _saved_eval_message(level, report, stale=stale)
                    outputs = _evaluation_outputs(ws, report, status)
                    return (sid, outputs[-1], *outputs[:-1])

                def run_eval(sid, key, model_name, level, quota_safe, target_rpm, reuse_saved, include_profiles, request: gr.Request):
                    client = getattr(getattr(request, "client", None), "host", None) or "unknown"
                    try:
                        limiter.check(f"ui-eval:{client}")
                        sid, ws = _ensure_session(sid)
                        if not ws.chunks:
                            ws.ingest(_demo_paths(), ocr=False, api_key=(key or None), model=model_name)

                        benchmark_version = str(demo_benchmark_metadata().get("version", ""))
                        cached = ws.get_evaluation(
                            level,
                            model=model_name,
                            benchmark_version=benchmark_version,
                            require_current_corpus=True,
                        )
                        if reuse_saved and cached and (not include_profiles or bool(cached.get("profile_benchmark"))):
                            outputs = _evaluation_outputs(
                                ws,
                                cached,
                                _saved_eval_message(level, cached, stale=False),
                            )
                            return (
                                sid,
                                gr.Button(value="Run evaluation", interactive=True),
                                level,
                                *outputs,
                            )

                        standard_base = None
                        if level == "Deep" and reuse_saved:
                            standard_base = ws.get_evaluation(
                                "Standard",
                                model=model_name,
                                benchmark_version=benchmark_version,
                                require_current_corpus=True,
                            )

                        report = run_demo_eval(
                            ws,
                            key or None,
                            model_name,
                            level=level,
                            target_rpm=int(target_rpm) if quota_safe else 0,
                            base_standard_report=standard_base,
                            include_profile_benchmark=bool(include_profiles),
                        )
                        report = ws.save_evaluation(
                            level,
                            report,
                            model=model_name,
                            benchmark_version=benchmark_version,
                        )
                        skipped_note = (
                            " Quick mode intentionally skips the retrieval/context/compression/scale ablations and Deep judge."
                            if level == "Quick"
                            else ""
                        )
                        incremental_note = (
                            " Deep reused the saved Standard deterministic baseline and only ran sampled judge calls."
                            if report.get("summary", {}).get("reused_standard_baseline")
                            else ""
                        )
                        meta = report.get("evaluation_cache", {})
                        run_id = meta.get("run_id") or "unknown"
                        requests = int(report.get("summary", {}).get("gemini_requests", 0) or 0)
                        outputs = _evaluation_outputs(
                            ws,
                            report,
                            f"**Fresh {level} evaluation complete - run `{run_id}`.** "
                            f"This execution issued {requests} Gemini request(s) and was saved for reuse."
                            f"{skipped_note}{incremental_note}",
                        )
                        return (
                            sid,
                            gr.Button(value="Run evaluation", interactive=True),
                            level,
                            *outputs,
                        )
                    except Exception as exc:
                        error_status = (
                            "**Evaluation paused by Gemini quota.** The provider still returned a 429 after bounded "
                            "backoff. Leave quota-safe pacing enabled, lower the target RPM, or wait for the quota "
                            "window to reset.\n\n" + f"`{type(exc).__name__}: {exc}`"
                            if "429" in str(exc) or "quota" in str(exc).lower()
                            else f"**Evaluation failed.** `{type(exc).__name__}: {exc}`"
                        )
                        if 'ws' not in locals():
                            _, ws = _ensure_session(sid or None)
                        empty_outputs = _evaluation_outputs(ws, {}, error_status)
                        return (
                            sid or ws.session_id,
                            gr.Button(value="Run evaluation", interactive=True),
                            level,
                            *empty_outputs,
                        )

                def prepare_eval_table_export(sid, level, table_label, export_format):
                    sid, ws = _ensure_session(sid)
                    report = ws.get_evaluation(level, require_current_corpus=False)
                    if not report:
                        return (
                            "",
                            None,
                            f"**No saved {level} evaluation exists.** Run or load that evaluation first.",
                        )
                    report = to_jsonable(report)
                    frame = _eval_table_frame(ws, report, table_label)
                    if frame.empty:
                        return (
                            "",
                            None,
                            f"**{table_label} is empty for the saved {level} evaluation.**",
                        )
                    content, extension = _table_export_text(frame, export_format)
                    export_dir = ws.evaluation_dir / "exports"
                    export_dir.mkdir(parents=True, exist_ok=True)
                    safe_table = re.sub(r"[^a-z0-9]+", "_", table_label.lower()).strip("_") or "table"
                    safe_level = re.sub(r"[^a-z0-9]+", "_", level.lower()).strip("_") or "evaluation"
                    target = export_dir / f"{safe_level}_{safe_table}.{extension}"
                    target.write_text(content, encoding="utf-8")
                    return (
                        content,
                        str(target),
                        f"**Prepared {table_label} from the saved {level} run as {export_format}.** "
                        "Use the copy button in the code box or Download table.",
                    )

                eval_event = eval_btn.click(
                    begin_eval,
                    [eval_level, eval_quota_safe, eval_target_rpm, eval_profile_benchmark],
                    [eval_btn, eval_status, eval_scorecard],
                    queue=False,
                    show_progress="hidden",
                )
                eval_event.then(
                    run_eval,
                    [
                        session_state, api_key, model, eval_level, eval_quota_safe, eval_target_rpm,
                        eval_reuse_saved, eval_profile_benchmark,
                    ],
                    [
                        session_state, eval_btn, eval_saved_level, eval_scorecard, eval_diagnostics,
                        eval_qa, eval_planner, eval_overview, eval_sql, eval_ablation, eval_context_budget,
                        eval_compression, eval_scale_stress, eval_readiness, eval_hard, eval_profiles,
                        eval_node_latency, eval_abstention, eval_compare, eval_history, eval_output, eval_saved_status, eval_status,
                    ],
                    show_progress="hidden",
                )

                eval_export_btn.click(
                    prepare_eval_table_export,
                    [session_state, eval_saved_level, eval_export_table, eval_export_format],
                    [eval_export_code, eval_export_download, eval_export_status],
                    queue=False,
                    show_progress="hidden",
                )

                eval_saved_level.input(
                    load_saved_eval,
                    [session_state, eval_saved_level],
                    [
                        session_state, eval_status, eval_scorecard, eval_diagnostics,
                        eval_qa, eval_planner, eval_overview, eval_sql, eval_ablation, eval_context_budget,
                        eval_compression, eval_scale_stress, eval_readiness, eval_hard, eval_profiles,
                        eval_node_latency, eval_abstention, eval_compare, eval_history, eval_output, eval_saved_status,
                    ],
                    queue=False,
                    show_progress="hidden",
                )
                eval_refresh_saved.click(
                    load_saved_eval,
                    [session_state, eval_saved_level],
                    [
                        session_state, eval_status, eval_scorecard, eval_diagnostics,
                        eval_qa, eval_planner, eval_overview, eval_sql, eval_ablation, eval_context_budget,
                        eval_compression, eval_scale_stress, eval_readiness, eval_hard, eval_profiles,
                        eval_node_latency, eval_abstention, eval_compare, eval_history, eval_output, eval_saved_status,
                    ],
                    queue=False,
                    show_progress="hidden",
                )

            with gr.Tab("Architecture + API"):
                gr.Markdown(
                    "### System architecture and API\n"
                    "Inspect the live workspace, LangGraph responsibilities, REST surface and copy-ready request examples."
                )
                arch_refresh = gr.Button("Refresh runtime view")
                arch_runtime = gr.Markdown(
                    "Click **Refresh runtime view** to show the current session, corpus and model configuration.",
                    elem_classes=["status-line"],
                )

                with gr.Tabs():
                    with gr.Tab("Pipeline architecture"):
                        gr.Markdown(
                            """
**End-to-end flow**

`Upload/ZIP -> secure parsing -> chunk index + source-profile index -> semantic planner -> task-aware retrieval -> adaptive reranker/context budget -> evidence grading -> corrective retrieval -> conditional Ask-the-Web -> grounded Gemini generation -> Self-RAG verification -> cited answer`

**Retrieval choices**
- `global` - source-balanced corpus overview
- `hierarchical` - source selection followed by within-source chunk retrieval
- `semantic` - focused dense + BM25 hybrid retrieval
- `analytical` - source-balanced document evidence plus deterministic table evidence for trend/insight synthesis
- `table` - read-only DuckDB Text2SQL
- `none` - external-only/web task

Weak local retrieval does not automatically trigger the web. CRAG first corrects local retrieval and only uses external search when the semantic plan says external knowledge is relevant.
                            """
                        )
                        gr.Dataframe(value=_pipeline_stage_frame(), interactive=False, wrap=True, label="LangGraph nodes")

                    with gr.Tab("REST API"):
                        gr.Markdown(
                            "FastAPI exposes interactive Swagger documentation at `/docs` and the OpenAPI schema at `/openapi.json`. "
                            "If `APP_API_TOKEN` is configured, protected endpoints require `Authorization: Bearer <token>`."
                        )
                        gr.Dataframe(value=_api_endpoint_frame(), interactive=False, wrap=True, label="Endpoint reference")
                        api_examples = gr.Code(
                            value="# Refresh the runtime view to generate examples for the current workspace.",
                            label="Copy-ready curl examples",
                        )

                    with gr.Tab("Runtime snapshot"):
                        arch_json = gr.JSON(label="Current runtime and workspace")
                        gr.Markdown(
                            "Runtime indexes are ephemeral on standard Hugging Face Space storage. Browser state can restore a "
                            "session ID across a normal refresh, but custom uploads must be re-indexed after a container restart."
                        )

                    with gr.Tab("Evaluation architecture"):
                        gr.Markdown(
                            """
The bundled benchmark evaluates separate failure surfaces rather than relying on one opaque score:

- retrieval - source Hit@1, Recall@5, MRR, AP@5, nDCG@5 and duplicate-source rate
- orchestration - route/task/strategy accuracy and web-use precision/recall
- generation - answer-key checks, citation validity and citation coverage
- structured data - Text2SQL routing plus computed-answer checks
- lifecycle - zero-call abstention for missing local resources
- efficiency - cache-bypassed pipeline latency, planner latency, LLM-call estimate and reranker ablation
- Deep mode - calibrated Gemini judge whose citation score cannot override deterministic citation failures
- saved runs - Quick/Standard/Deep reports are kept per workspace with corpus/model/benchmark metadata
- incremental Deep - a compatible Standard baseline can be reused so Deep adds only the sampled judge layer

Detailed metrics and acceptance checks are shown separately so a single aggregate score does not hide subsystem behavior.
                            """
                        )

                arch_refresh.click(
                    _architecture_snapshot,
                    [session_state],
                    [session_state, arch_runtime, api_examples, arch_json],
                    queue=False,
                    show_progress="hidden",
                )

        gr.HTML(
            '<div class="footer-note">RAGForge - document, table, and web retrieval with inspectable sources and evaluation.</div>'
        )
        return demo
