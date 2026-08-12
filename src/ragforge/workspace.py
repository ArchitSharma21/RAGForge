from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .chunking import chunk_documents
from .config import get_settings
from .context_budget import corpus_scale_label
from .corpus import build_source_profiles, corpus_manifest, profile_chunks
from .llm import GeminiGateway
from .json_utils import to_jsonable
from .loaders import DocumentLoader
from .retrieval import HybridRetriever
from .schemas import CorpusSummary, Document, SearchHit, SourceProfile
from .sql_agent import SQLWorkspace


SERVER_BOOT_ID = uuid.uuid4().hex[:12]


class Workspace:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = time.time()
        self.last_access = self.created_at
        self.version = 0
        self.settings = get_settings()
        self.dir = self.settings.data_dir / session_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.documents: list[Document] = []
        self.sources: list[str] = []
        self.chunks = []
        self.retriever = HybridRetriever("chunks")
        self.source_profiles: dict[str, SourceProfile] = {}
        self.source_retriever = HybridRetriever("source_profiles")
        self.sql = SQLWorkspace()
        self.history: list[dict[str, str]] = []
        self.ingested_hashes: set[str] = set()
        self.evaluation_reports: dict[str, dict[str, Any]] = {}
        self.lock = threading.RLock()

    def touch(self) -> None:
        self.last_access = time.time()

    def ingest(
        self,
        paths: list[Path],
        ocr: bool = False,
        semantic_chunking: bool = False,
        api_key: str | None = None,
        model: str | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> CorpusSummary:
        with self.lock:
            self.touch()
            notify = progress_callback or (lambda _progress, _message: None)
            notify(0.02, "Preparing corpus inputs")
            gateway = GeminiGateway(api_key, model) if ocr else None
            loader = DocumentLoader(gateway)
            expanded = loader.expand_inputs(paths, self.dir)
            notify(0.08, f"Found {len(expanded)} supported file(s)")
            new_docs: list[Document] = []
            total_files = max(1, len(expanded))
            for file_idx, path in enumerate(expanded, start=1):
                notify(0.10 + 0.42 * (file_idx - 1) / total_files, f"Parsing {path.name} ({file_idx}/{len(expanded)})")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in self.ingested_hashes:
                    continue
                docs, tables = loader.load(path)
                new_docs.extend(docs)
                for name, df in tables:
                    self.sql.add_dataframe(name, df)
                self.ingested_hashes.add(digest)

            notify(0.54, "Chunking documents")
            self.documents.extend(new_docs)
            self.sources = sorted(set(self.sources + [d.source for d in new_docs]))
            self.chunks = chunk_documents(self.documents, semantic=semantic_chunking)
            notify(0.66, f"Building hybrid chunk index ({len(self.chunks)} chunks)")
            self.retriever.index(self.chunks)

            # Build a second, source-level representation. This powers semantic
            # source selection and corpus overviews without allowing a 48-page
            # PDF to swamp a five-file corpus simply because it produced more
            # chunks than the other sources.
            notify(0.82, "Building source profiles")
            self.source_profiles = build_source_profiles(self.documents, self.chunks)
            notify(0.92, f"Building source index ({len(self.source_profiles)} sources)")
            self.source_retriever.index(profile_chunks(self.source_profiles))
            self.version += 1
            notify(1.0, "Corpus ready")
            return self.summary()


    @property
    def is_empty(self) -> bool:
        return not self.chunks and not self.sql.tables

    def stats(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "version": self.version,
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "source_profiles": len(self.source_profiles),
            "sources": len(self.sources),
            "tables": len(self.sql.tables),
            "table_names": list(self.sql.tables),
            "saved_evaluations": sorted(self.evaluation_reports),
            "status": "empty" if self.is_empty else "ready",
            "corpus_scale": corpus_scale_label(len(self.chunks), len(self.sources)),
            "server_boot_id": SERVER_BOOT_ID,
        }

    def health_snapshot(self) -> dict[str, object]:
        """Operational snapshot for the Architecture/API runtime view."""
        now = time.time()
        chunk_chars = sum(len(getattr(chunk, "text", "") or "") for chunk in self.chunks)
        vector_bytes = int(getattr(getattr(self.retriever, "_vectors", None), "nbytes", 0) or 0)
        source_vector_bytes = int(getattr(getattr(self.source_retriever, "_vectors", None), "nbytes", 0) or 0)
        text_bytes = chunk_chars
        estimated_index_mb = (vector_bytes + source_vector_bytes + text_bytes) / (1024 * 1024)
        max_chunks = max(1, int(self.settings.max_chunks_per_session))
        utilization = len(self.chunks) / max_chunks
        return {
            **self.stats(),
            "session_age_minutes": round((now - self.created_at) / 60.0, 1),
            "idle_minutes": round((now - self.last_access) / 60.0, 1),
            "session_ttl_minutes": int(self.settings.session_ttl_minutes),
            "max_chunks_per_session": max_chunks,
            "chunk_capacity_utilization": round(utilization, 3),
            "capacity_status": "warning" if utilization >= 0.80 else "ok",
            "chunk_text_chars": chunk_chars,
            "vector_bytes": vector_bytes,
            "source_vector_bytes": source_vector_bytes,
            "estimated_index_memory_mb": round(estimated_index_mb, 2),
            "chunk_index_ready": bool(getattr(self.retriever, "_ready", False)),
            "source_index_ready": bool(getattr(self.source_retriever, "_ready", False)),
            "evaluation_history_runs": len(self.evaluation_history_inventory()),
            "adaptive_policy": {
                "adaptive_top_k": True,
                "focused_context_budget": "2-5 chunks based on scale/confidence",
                "focused_evidence_compression": True,
                "small_corpus_reranker_skip": len(self.chunks) < 250 and len(self.sources) < 10,
            },
        }

    @property
    def evaluation_dir(self) -> Path:
        path = self.dir / "evaluations"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def evaluation_history_dir(self) -> Path:
        path = self.evaluation_dir / "history"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_evaluation(
        self,
        level: str,
        report: dict[str, Any],
        *,
        model: str,
        benchmark_version: str,
    ) -> dict[str, Any]:
        """Persist the latest evaluation for one depth within this workspace.

        Evaluation reports are deliberately separate from the response cache.
        They survive browser refreshes while the Hugging Face container is
        alive, but remain ephemeral with the rest of the workspace storage.
        """
        with self.lock:
            self.touch()
            saved = to_jsonable(report)
            saved["evaluation_cache"] = {
                "level": level,
                "model": model,
                "benchmark_version": benchmark_version,
                "workspace_version": self.version,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "run_id": uuid.uuid4().hex[:12],
                "server_boot_id": SERVER_BOOT_ID,
            }
            self.evaluation_reports[level] = saved
            try:
                target = self.evaluation_dir / f"{level.lower()}.json"
                target.write_text(json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8")
                stamp = saved["evaluation_cache"]["saved_at"].replace(":", "-").replace("+", "_")
                archive = self.evaluation_history_dir / f"{stamp}_{level.lower()}.json"
                archive.write_text(json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                # In-memory history is still useful even if persistence fails.
                pass
            return saved

    def get_evaluation(
        self,
        level: str,
        *,
        model: str | None = None,
        benchmark_version: str | None = None,
        require_current_corpus: bool = True,
    ) -> dict[str, Any] | None:
        with self.lock:
            self.touch()
            report = self.evaluation_reports.get(level)
            if report is None:
                path = self.evaluation_dir / f"{level.lower()}.json"
                if path.exists():
                    try:
                        report = to_jsonable(json.loads(path.read_text(encoding="utf-8")))
                        self.evaluation_reports[level] = report
                    except Exception:
                        report = None
            if not report:
                return None
            report = to_jsonable(report)
            self.evaluation_reports[level] = report
            meta = report.get("evaluation_cache", {})
            if require_current_corpus and int(meta.get("workspace_version", -1)) != int(self.version):
                return None
            if model and meta.get("model") != model:
                return None
            if benchmark_version and meta.get("benchmark_version") != benchmark_version:
                return None
            return report

    def evaluation_inventory(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for level in ("Quick", "Standard", "Deep"):
            report = self.get_evaluation(level, require_current_corpus=False)
            if not report:
                continue
            meta = report.get("evaluation_cache", {})
            summary = report.get("summary", {})
            rows.append(
                {
                    "level": level,
                    "grade": summary.get("quality_grade", "-"),
                    "score": summary.get("deterministic_quality_score"),
                    "model": meta.get("model", "-"),
                    "benchmark": meta.get("benchmark_version", "-"),
                    "workspace_version": meta.get("workspace_version"),
                    "current_corpus": int(meta.get("workspace_version", -1)) == int(self.version),
                    "saved_at": meta.get("saved_at", ""),
                    "run_id": meta.get("run_id", ""),
                    "server_boot_id": meta.get("server_boot_id", ""),
                    "current_server": meta.get("server_boot_id") in {None, "", SERVER_BOOT_ID},
                }
            )
        return rows

    def evaluation_history_inventory(self, limit: int = 50) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            paths = sorted(self.evaluation_history_dir.glob("*.json"))[-max(1, int(limit)) :]
        except Exception:
            paths = []
        previous_by_level: dict[str, dict[str, Any]] = {}
        for path in paths:
            try:
                report = to_jsonable(json.loads(path.read_text(encoding="utf-8")))
                meta = report.get("evaluation_cache", {})
                summary = report.get("summary", {})
                level = str(meta.get("level", summary.get("evaluation_level", "-")))
                row = {
                    "saved_at": meta.get("saved_at", ""),
                    "level": level,
                    "benchmark": meta.get("benchmark_version", summary.get("benchmark_version", "-")),
                    "model": meta.get("model", "-"),
                    "workspace_version": meta.get("workspace_version"),
                    "grade": summary.get("quality_grade", "-"),
                    "score": summary.get("deterministic_quality_score"),
                    "citation_coverage": summary.get("citation_coverage"),
                    "hard_mode_pass": summary.get("hard_mode_pass_rate"),
                    "p50_ms": summary.get("latency_p50_ms"),
                    "gemini_requests": summary.get("gemini_requests"),
                    "run_id": meta.get("run_id", ""),
                    "server_boot_id": meta.get("server_boot_id", ""),
                }
                prev = previous_by_level.get(level)
                if prev:
                    try:
                        row["delta_score"] = round(float(row.get("score") or 0) - float(prev.get("score") or 0), 3)
                        row["delta_p50_ms"] = round(float(row.get("p50_ms") or 0) - float(prev.get("p50_ms") or 0), 1)
                    except Exception:
                        row["delta_score"] = None
                        row["delta_p50_ms"] = None
                else:
                    row["delta_score"] = None
                    row["delta_p50_ms"] = None
                previous_by_level[level] = row
                rows.append(row)
            except Exception:
                continue
        return rows

    def manifest(self, max_chars: int = 9000, include_excerpts: bool = True) -> str:
        base = corpus_manifest(
            self.source_profiles,
            list(self.sql.tables),
            max_chars=max_chars,
            include_excerpts=include_excerpts,
        )
        schema = self.sql.schema_text().strip()
        if schema:
            base += "\nStructured table schemas:\n" + schema
        return base[:max_chars]

    def select_source_hits(self, query: str, limit: int = 5) -> list[SearchHit]:
        if not self.source_profiles:
            return []
        return self.source_retriever.search(
            query,
            top_k=min(max(1, limit), len(self.source_profiles)),
            use_reranker=False,
        )

    def select_sources(self, query: str, limit: int = 5) -> list[str]:
        hits = self.select_source_hits(query, limit)
        sources: list[str] = []
        for hit in hits:
            if hit.chunk.source not in sources:
                sources.append(hit.chunk.source)
        return sources

    def global_evidence(self, query: str, top_k: int, use_reranker: bool = True) -> list[SearchHit]:
        """Stable source-balanced evidence for corpus overview/synthesis tasks.

        Source profiles decide *which sources* matter. Evidence then comes from
        deterministic representative original chunks, rather than asking an
        abstract overview query to choose an arbitrary page from every source.
        """
        if not self.source_profiles:
            return []
        source_limit = min(max(1, top_k), len(self.source_profiles))
        source_hits = self.select_source_hits(query, source_limit)
        if not source_hits:
            return []

        selected: list[SearchHit] = []
        round_idx = 0
        while len(selected) < top_k:
            added = False
            for source_hit in source_hits:
                profile = self.source_profiles.get(source_hit.chunk.source)
                if not profile or round_idx >= len(profile.representative_chunk_ids):
                    continue
                chunk_id = profile.representative_chunk_ids[round_idx]
                chunk = self.retriever.chunk_by_id.get(chunk_id)
                if not chunk:
                    continue
                decay = 1.0 / (1.0 + 0.15 * round_idx)
                selected.append(
                    SearchHit(
                        chunk=chunk,
                        score=float(source_hit.score) * decay,
                        dense_score=(float(source_hit.dense_score) * decay if source_hit.dense_score is not None else None),
                        sparse_score=(float(source_hit.sparse_score) * decay if source_hit.sparse_score is not None else None),
                    )
                )
                added = True
                if len(selected) >= top_k:
                    break
            if not added:
                break
            round_idx += 1

        # Reranking can improve ordering, but keep the source-balanced selection
        # itself intact. Raw reranker logits are never used as confidence.
        if use_reranker and selected:
            try:
                from .retrieval import ModelRegistry

                scores = list(ModelRegistry.reranker().rerank(query, [h.chunk.text for h in selected]))
                for hit, score in zip(selected, scores):
                    hit.rerank_score = float(score)
            except Exception:
                pass
        return selected[:top_k]

    def hierarchical_evidence(
        self,
        query: str,
        top_k: int,
        use_reranker: bool = True,
        source_limit: int = 4,
        diversify: bool = False,
    ) -> tuple[list[SearchHit], list[str]]:
        """Retrieve source profiles first, then chunks only from selected sources."""
        selected_sources = self.select_sources(query, min(source_limit, max(1, len(self.source_profiles))))
        hits = self.retriever.search(
            query,
            top_k=max(top_k * 2, 8),
            use_reranker=use_reranker,
            allowed_sources=selected_sources,
        )
        if diversify:
            # Re-run a source-balanced evidence pass for synthesis/comparison so
            # multiple selected sources are represented when relevant.
            diverse = self.retriever.source_balanced_search(
                query,
                top_k=top_k,
                sources=selected_sources,
                per_source=1,
                use_reranker=use_reranker,
            )
            if diverse:
                hits = diverse
        return hits[:top_k], selected_sources

    def reset(self) -> None:
        try:
            shutil.rmtree(self.dir, ignore_errors=True)
        finally:
            self.__init__(self.session_id)

    def summary(self) -> CorpusSummary:
        return CorpusSummary(
            session_id=self.session_id,
            documents=len(self.documents),
            chunks=len(self.chunks),
            tables=list(self.sql.tables),
            sources=list(self.sources),
            source_profiles=len(self.source_profiles),
        )


class WorkspaceRegistry:
    def __init__(self):
        self.settings = get_settings()
        self._items: dict[str, Workspace] = {}
        self._lock = threading.RLock()

    def create(self) -> Workspace:
        with self._lock:
            self.cleanup()
            session_id = uuid.uuid4().hex
            ws = Workspace(session_id)
            self._items[session_id] = ws
            return ws

    def get(self, session_id: str | None) -> Workspace:
        """UI-friendly lookup: return an existing workspace or create a fresh one."""
        with self._lock:
            self.cleanup()
            if session_id and session_id in self._items:
                ws = self._items[session_id]
                ws.touch()
                return ws
            return self.create()

    def contains(self, session_id: str | None) -> bool:
        with self._lock:
            self.cleanup()
            return bool(session_id and session_id in self._items)

    def require(self, session_id: str) -> Workspace:
        """API lookup: never silently replace a missing/expired client session id."""
        with self._lock:
            self.cleanup()
            ws = self._items.get(session_id)
            if ws is None:
                raise KeyError("Unknown or expired session_id; create a new session first")
            ws.touch()
            return ws

    def delete(self, session_id: str) -> None:
        with self._lock:
            ws = self._items.pop(session_id, None)
            if ws:
                shutil.rmtree(ws.dir, ignore_errors=True)

    def cleanup(self) -> None:
        cutoff = time.time() - self.settings.session_ttl_minutes * 60
        stale = [sid for sid, ws in self._items.items() if ws.last_access < cutoff]
        for sid in stale:
            ws = self._items.pop(sid)
            shutil.rmtree(ws.dir, ignore_errors=True)


registry = WorkspaceRegistry()
