from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from .config import get_settings
from .evaluation import demo_benchmark_metadata, run_demo_eval
from .pipeline import RAGEngine
from .rate_limit import RateLimitExceeded, limiter
from .schemas import CorpusSummary, EvaluationRequest, QueryRequest, QueryResponse, SessionResponse
from .workspace import registry

REQUESTS = Counter("ragforge_requests_total", "Total API requests", ["endpoint", "status"])
LATENCY = Histogram("ragforge_request_latency_seconds", "API request latency", ["endpoint"])


def _auth(authorization: Annotated[str | None, Header()] = None) -> None:
    token = get_settings().app_api_token
    if not token:
        return
    if authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="Invalid or missing Bearer token")


def create_api() -> FastAPI:
    app = FastAPI(title="RAGForge API", version="2.0.3")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "service": "RAGForge", "model": get_settings().default_model}

    @app.get("/api/v1/info")
    def info():
        s = get_settings()
        return {
            "app": s.app_name,
            "default_model": s.default_model,
            "embedding_model": s.embedding_model,
            "reranker_model": s.reranker_model,
            "native_search_model": s.native_search_model,
            "features": [
                "semantic-query-planning", "source-profile-index", "hierarchical-retrieval",
                "source-balanced-global-retrieval", "hybrid-search", "reranking", "hyde",
                "corrective-rag", "conditional-web", "self-rag", "text2sql", "ask-the-web",
                "citations", "guardrails", "layered-evaluation", "retrieval-ablation", "quality-gated-evaluation",
                "cache-bypassed-benchmarking", "quota-aware-evaluation", "retry-after-backoff",
                "workspace-preflight", "browser-session-continuity", "lazy-demo-recovery", "explicit-abstention",
                "saved-evaluation-history", "incremental-deep-evaluation", "typed-text2sql-evaluation",
                "adaptive-reranking", "semantic-citation-attribution", "insight-synthesis",
                "table-citations", "hard-mode-evaluation", "chunk-level-reranker-ablation",
                "optional-profile-benchmark", "node-latency-observability", "evaluation-run-history",
                "grounded-absence-handling", "markdown-aware-citation-coverage", "evaluation-run-provenance",
                "profile-policy-diagnostics", "context-efficiency-diagnostics",
                "focused-context-pruning", "adaptive-context-budgeting", "adaptive-retrieval-depth",
                "focused-evidence-compression", "context-budget-ablation", "compression-ablation",
                "scale-stress-evaluation", "release-readiness-checklist", "prompt-budget-observability",
                "retrieval-confidence-telemetry", "workspace-health-diagnostics"
            ],
        }

    @app.post("/api/v1/session", response_model=SessionResponse, dependencies=[Depends(_auth)])
    def new_session():
        ws = registry.create()
        return SessionResponse(session_id=ws.session_id)

    @app.get("/api/v1/session/{session_id}", dependencies=[Depends(_auth)])
    def session_status(session_id: str):
        try:
            return registry.require(session_id).stats()
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/session/{session_id}/diagnostics", dependencies=[Depends(_auth)])
    def session_diagnostics(session_id: str):
        try:
            return registry.require(session_id).health_snapshot()
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/evaluation/benchmark")
    def evaluation_benchmark():
        return demo_benchmark_metadata()

    @app.get("/api/v1/evaluation/saved/{session_id}", dependencies=[Depends(_auth)])
    def saved_evaluations(session_id: str):
        try:
            return {"session_id": session_id, "runs": registry.require(session_id).evaluation_inventory()}
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/evaluation/history/{session_id}", dependencies=[Depends(_auth)])
    def evaluation_history(session_id: str):
        try:
            return {"session_id": session_id, "runs": registry.require(session_id).evaluation_history_inventory()}
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/evaluation/saved/{session_id}/{level}", dependencies=[Depends(_auth)])
    def saved_evaluation(session_id: str, level: str):
        normalized = level.strip().title()
        if normalized not in {"Quick", "Standard", "Deep"}:
            raise HTTPException(status_code=400, detail="level must be Quick, Standard or Deep")
        try:
            ws = registry.require(session_id)
            report = ws.get_evaluation(normalized, require_current_corpus=False)
            if not report:
                raise HTTPException(status_code=404, detail=f"No saved {normalized} evaluation")
            return report
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/v1/ingest", response_model=CorpusSummary, dependencies=[Depends(_auth)])
    async def ingest(
        request: Request,
        session_id: Annotated[str, Form()],
        files: Annotated[list[UploadFile], File()],
        use_ocr: Annotated[bool, Form()] = False,
        semantic_chunking: Annotated[bool, Form()] = False,
    ):
        started = time.perf_counter()
        try:
            client = request.client.host if request.client else "unknown"
            limiter.check(f"api-ingest:{client}")
            ws = registry.require(session_id)
            paths = []
            max_bytes = get_settings().max_upload_mb * 1024 * 1024
            with tempfile.TemporaryDirectory() as td:
                for upload in files:
                    target = Path(td) / Path(upload.filename or "upload").name
                    written = 0
                    with target.open("wb") as fh:
                        while True:
                            block = await upload.read(1024 * 1024)
                            if not block:
                                break
                            written += len(block)
                            if written > max_bytes:
                                raise ValueError(f"{target.name} exceeds the configured upload limit")
                            fh.write(block)
                    paths.append(target)
                result = ws.ingest(paths, ocr=use_ocr, semantic_chunking=semantic_chunking)
            REQUESTS.labels("ingest", "ok").inc()
            return result
        except RateLimitExceeded as exc:
            REQUESTS.labels("ingest", "rate_limited").inc()
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except Exception as exc:
            REQUESTS.labels("ingest", "error").inc()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            LATENCY.labels("ingest").observe(time.perf_counter() - started)

    @app.post("/api/v1/query", response_model=QueryResponse, dependencies=[Depends(_auth)])
    def query(payload: QueryRequest, request: Request):
        started = time.perf_counter()
        try:
            client = request.client.host if request.client else "unknown"
            limiter.check(f"api:{client}")
            ws = registry.require(payload.session_id)
            result = RAGEngine(ws).ask(payload.query, payload.config)
            REQUESTS.labels("query", "ok").inc()
            return result
        except RateLimitExceeded as exc:
            REQUESTS.labels("query", "rate_limited").inc()
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except Exception as exc:
            REQUESTS.labels("query", "error").inc()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            LATENCY.labels("query").observe(time.perf_counter() - started)

    @app.post("/api/v1/evaluate/demo", dependencies=[Depends(_auth)])
    def evaluate_demo(payload: EvaluationRequest, request: Request):
        started = time.perf_counter()
        try:
            client = request.client.host if request.client else "unknown"
            limiter.check(f"api-eval:{client}")
            ws = registry.require(payload.session_id)
            required_demo_sources = {
                "acme_cloud_runbook.md",
                "orbitpay_policy.txt",
                "release_notes.html",
                "support_matrix.csv",
                "NIST_AI_RMF_1.0.pdf",
            }
            if not required_demo_sources.issubset(set(ws.sources)):
                raise ValueError(
                    "The bundled demo benchmark requires the five bundled demo sources to be indexed in this session."
                )
            benchmark_version = str(demo_benchmark_metadata().get("version", ""))
            if payload.reuse_saved:
                cached = ws.get_evaluation(
                    payload.level,
                    model=payload.model,
                    benchmark_version=benchmark_version,
                    require_current_corpus=True,
                )
                if cached:
                    REQUESTS.labels("evaluate_demo", "ok").inc()
                    return cached

            standard_base = None
            if payload.level == "Deep" and payload.reuse_saved and not payload.include_profile_benchmark:
                standard_base = ws.get_evaluation(
                    "Standard",
                    model=payload.model,
                    benchmark_version=benchmark_version,
                    require_current_corpus=True,
                )

            report = run_demo_eval(
                ws,
                api_key=None,
                model=payload.model,
                level=payload.level,
                target_rpm=payload.target_rpm,
                base_standard_report=standard_base,
                include_profile_benchmark=payload.include_profile_benchmark,
            )
            report = ws.save_evaluation(
                payload.level,
                report,
                model=payload.model,
                benchmark_version=benchmark_version,
            )
            REQUESTS.labels("evaluate_demo", "ok").inc()
            return report
        except RateLimitExceeded as exc:
            REQUESTS.labels("evaluate_demo", "rate_limited").inc()
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except Exception as exc:
            REQUESTS.labels("evaluate_demo", "error").inc()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            LATENCY.labels("evaluate_demo").observe(time.perf_counter() - started)

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app
