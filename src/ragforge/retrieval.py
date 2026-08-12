from __future__ import annotations

import re
import threading
from collections import defaultdict
from collections.abc import Iterable

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

from .config import get_settings
from .schemas import Chunk, SearchHit


class ModelRegistry:
    _lock = threading.Lock()
    _embedding: TextEmbedding | None = None
    _reranker: TextCrossEncoder | None = None

    @classmethod
    def embedding(cls) -> TextEmbedding:
        if cls._embedding is None:
            with cls._lock:
                if cls._embedding is None:
                    cls._embedding = TextEmbedding(model_name=get_settings().embedding_model)
        return cls._embedding

    @classmethod
    def reranker(cls) -> TextCrossEncoder:
        if cls._reranker is None:
            with cls._lock:
                if cls._reranker is None:
                    cls._reranker = TextCrossEncoder(model_name=get_settings().reranker_model)
        return cls._reranker


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _norm01(score: float) -> float:
    # Cosine scores from BGE are usually positive for plausible text matches;
    # clip rather than pretend a reranker logit is a calibrated probability.
    return max(0.0, min(1.0, float(score)))


class HybridRetriever:
    """Dense + BM25 hybrid retriever with optional source-scoped search.

    Qdrant remains the primary unfiltered vector store. We additionally keep the
    normalized embedding matrix in memory so hierarchical retrieval can search
    only the source(s) selected by the source-level index without rebuilding a
    vector collection per document.
    """

    def __init__(self, collection: str = "chunks"):
        self.client = QdrantClient(":memory:")
        self.collection = collection
        self.chunks: list[Chunk] = []
        self.chunk_by_id: dict[str, Chunk] = {}
        self.bm25: BM25Okapi | None = None
        self._ready = False
        self._vectors: np.ndarray | None = None
        self._source_indices: dict[str, np.ndarray] = {}

    def index(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.chunk_by_id = {c.id: c for c in chunks}
        self.bm25 = BM25Okapi([_tokens(c.text) for c in chunks]) if chunks else None
        self._source_indices = {}
        if not chunks:
            self._ready = False
            self._vectors = None
            return

        embedding = ModelRegistry.embedding()
        vectors = np.asarray(list(embedding.passage_embed([c.text for c in chunks])), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
        self._vectors = vectors / norms
        size = int(vectors.shape[1])

        for source in {c.source for c in chunks}:
            self._source_indices[source] = np.asarray(
                [i for i, c in enumerate(chunks) if c.source == source], dtype=np.int32
            )

        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={"dense": models.VectorParams(size=size, distance=models.Distance.COSINE)},
        )
        points = [
            models.PointStruct(
                id=idx,
                vector={"dense": vector.tolist()},
                payload={"chunk_id": chunk.id, "source": chunk.source},
            )
            for idx, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
        self.client.upload_points(collection_name=self.collection, points=points)
        self._ready = True

    def index_precomputed(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Index chunks using caller-supplied vectors.

        This is primarily used by deterministic scale-stress evaluation, where
        existing corpus vectors are cloned for synthetic distractor copies. It
        avoids re-embedding hundreds of repeated chunks while exercising the
        real Qdrant + BM25 retrieval path.
        """
        if len(chunks) != int(getattr(vectors, "shape", [0])[0]):
            raise ValueError("chunks and vectors must have the same length")
        self.chunks = list(chunks)
        self.chunk_by_id = {c.id: c for c in chunks}
        self.bm25 = BM25Okapi([_tokens(c.text) for c in chunks]) if chunks else None
        self._source_indices = {}
        if not chunks:
            self._ready = False
            self._vectors = None
            return

        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
        self._vectors = matrix / norms
        size = int(self._vectors.shape[1])
        for source in {c.source for c in chunks}:
            self._source_indices[source] = np.asarray(
                [i for i, c in enumerate(chunks) if c.source == source], dtype=np.int32
            )

        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={"dense": models.VectorParams(size=size, distance=models.Distance.COSINE)},
        )
        points = [
            models.PointStruct(
                id=idx,
                vector={"dense": vector.tolist()},
                payload={"chunk_id": chunk.id, "source": chunk.source},
            )
            for idx, (chunk, vector) in enumerate(zip(chunks, self._vectors))
        ]
        self.client.upload_points(collection_name=self.collection, points=points)
        self._ready = True

    def search(
        self,
        query: str,
        top_k: int = 6,
        use_reranker: bool = True,
        allowed_sources: Iterable[str] | None = None,
    ) -> list[SearchHit]:
        if not self._ready or not self.chunks:
            return []
        allowed = set(allowed_sources or []) or None
        settings = get_settings()
        dense = self._dense(query, settings.top_k_dense, allowed)
        sparse = self._sparse(query, settings.top_k_sparse, allowed)
        fused = self._rrf(dense, sparse)
        candidates = fused[: max(top_k * 3, 12)]
        if use_reranker and candidates:
            self._rerank(query, candidates)
        return candidates[:top_k]

    def source_balanced_search(
        self,
        query: str,
        top_k: int,
        sources: Iterable[str] | None = None,
        per_source: int = 1,
        use_reranker: bool = True,
    ) -> list[SearchHit]:
        """Return query-relevant evidence while preventing a long source from monopolizing top-k.

        This is task-driven diversity, not a query-string rule. It is used for
        overview/cross-document plans where breadth across distinct sources is
        part of evidence sufficiency.
        """
        if not self._ready or self._vectors is None:
            return []
        allowed = list(sources or sorted(self._source_indices))
        if not allowed:
            return []

        query_vec = self._query_vector(query)
        sparse_scores = self._all_sparse_scores(query)
        sparse_max = float(np.max(sparse_scores)) if sparse_scores.size and np.max(sparse_scores) > 0 else 1.0

        selected: list[SearchHit] = []
        for source in allowed:
            idxs = self._source_indices.get(source)
            if idxs is None or not len(idxs):
                continue
            dense_scores = self._vectors[idxs] @ query_vec
            local_sparse = sparse_scores[idxs] / sparse_max if sparse_scores.size else np.zeros(len(idxs))
            # Dense is the more reliable signal for broad synthesis; BM25 gives
            # exact terminology a useful but bounded boost.
            combined = 0.75 * np.clip(dense_scores, 0.0, 1.0) + 0.25 * np.clip(local_sparse, 0.0, 1.0)
            order = np.argsort(combined)[::-1][: max(1, per_source)]
            for local_idx in order:
                absolute_idx = int(idxs[int(local_idx)])
                chunk = self.chunks[absolute_idx]
                selected.append(
                    SearchHit(
                        chunk=chunk,
                        score=float(combined[int(local_idx)]),
                        dense_score=_norm01(float(dense_scores[int(local_idx)])),
                        sparse_score=_norm01(float(local_sparse[int(local_idx)])),
                    )
                )

        selected.sort(key=lambda h: h.score, reverse=True)
        # Preserve at least one candidate per source before allowing a second
        # chunk from the same source.
        diversified = self._source_round_robin(selected, top_k)
        if use_reranker and diversified:
            self._rerank(query, diversified, preserve_source_diversity=True)
        return diversified[:top_k]

    def _query_vector(self, query: str) -> np.ndarray:
        vector = np.asarray(list(ModelRegistry.embedding().query_embed([query]))[0], dtype=np.float32)
        return vector / (np.linalg.norm(vector) + 1e-9)

    def _dense(self, query: str, k: int, allowed_sources: set[str] | None = None) -> list[SearchHit]:
        if allowed_sources:
            if self._vectors is None:
                return []
            query_vec = self._query_vector(query)
            arrays = [self._source_indices[s] for s in allowed_sources if s in self._source_indices]
            idxs = np.concatenate(arrays).astype(np.int32, copy=False) if arrays else np.asarray([], dtype=np.int32)
            if not len(idxs):
                return []
            scores = self._vectors[idxs] @ query_vec
            order = np.argsort(scores)[::-1][: min(k, len(idxs))]
            hits: list[SearchHit] = []
            for local_idx in order:
                idx = int(idxs[int(local_idx)])
                chunk = self.chunks[idx]
                score = _norm01(float(scores[int(local_idx)]))
                if float(chunk.metadata.get("injection_score", 0)) >= 0.5:
                    score *= 0.35
                hits.append(SearchHit(chunk=chunk, score=score, dense_score=score))
            return hits

        emb = self._query_vector(query)
        result = self.client.query_points(
            collection_name=self.collection,
            using="dense",
            query=emb.tolist(),
            with_payload=True,
            limit=min(k, len(self.chunks)),
        )
        hits: list[SearchHit] = []
        for point in result.points:
            chunk = self.chunk_by_id.get(point.payload.get("chunk_id"))
            if not chunk:
                continue
            score = _norm01(float(point.score))
            if float(chunk.metadata.get("injection_score", 0)) >= 0.5:
                score *= 0.35
            hits.append(SearchHit(chunk=chunk, score=score, dense_score=score))
        return hits

    def _all_sparse_scores(self, query: str) -> np.ndarray:
        if not self.bm25:
            return np.zeros(len(self.chunks), dtype=float)
        return np.asarray(self.bm25.get_scores(_tokens(query)), dtype=float)

    def _sparse(self, query: str, k: int, allowed_sources: set[str] | None = None) -> list[SearchHit]:
        scores = self._all_sparse_scores(query)
        if not len(scores):
            return []
        if allowed_sources:
            valid = np.asarray([c.source in allowed_sources for c in self.chunks], dtype=bool)
            scores = np.where(valid, scores, -np.inf)
        finite = np.isfinite(scores)
        if not finite.any():
            return []
        idxs = np.argsort(scores)[::-1][: min(k, int(finite.sum()))]
        positive = [int(i) for i in idxs if np.isfinite(scores[int(i)]) and scores[int(i)] > 0]
        if not positive:
            return []
        max_score = max(float(scores[i]) for i in positive) or 1.0
        hits: list[SearchHit] = []
        for idx in positive:
            norm = float(scores[idx]) / max_score
            chunk = self.chunks[idx]
            if float(chunk.metadata.get("injection_score", 0)) >= 0.5:
                norm *= 0.35
            hits.append(SearchHit(chunk=chunk, score=norm, sparse_score=norm))
        return hits

    def _rerank(self, query: str, hits: list[SearchHit], preserve_source_diversity: bool = False) -> None:
        try:
            scores = list(ModelRegistry.reranker().rerank(query, [hit.chunk.text for hit in hits]))
            for hit, score in zip(hits, scores):
                hit.rerank_score = float(score)
            hits.sort(key=lambda h: h.rerank_score if h.rerank_score is not None else -999.0, reverse=True)
            if preserve_source_diversity:
                hits[:] = self._source_round_robin(hits, len(hits))
        except Exception:
            return

    @staticmethod
    def _source_round_robin(hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        if not hits:
            return []
        by_source: dict[str, list[SearchHit]] = defaultdict(list)
        source_order: list[str] = []
        for hit in hits:
            if hit.chunk.source not in by_source:
                source_order.append(hit.chunk.source)
            by_source[hit.chunk.source].append(hit)

        out: list[SearchHit] = []
        round_idx = 0
        while len(out) < top_k:
            added = False
            for source in source_order:
                bucket = by_source[source]
                if round_idx < len(bucket):
                    out.append(bucket[round_idx])
                    added = True
                    if len(out) >= top_k:
                        break
            if not added:
                break
            round_idx += 1
        return out

    def _rrf(self, dense: list[SearchHit], sparse: list[SearchHit], k: int = 60) -> list[SearchHit]:
        scores: dict[str, float] = defaultdict(float)
        records: dict[str, SearchHit] = {}
        for ranking in (dense, sparse):
            for rank, hit in enumerate(ranking, start=1):
                scores[hit.chunk.id] += 1.0 / (k + rank)
                if hit.chunk.id not in records:
                    records[hit.chunk.id] = hit
                else:
                    records[hit.chunk.id].dense_score = records[hit.chunk.id].dense_score or hit.dense_score
                    records[hit.chunk.id].sparse_score = records[hit.chunk.id].sparse_score or hit.sparse_score
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        max_rrf = ordered[0][1] if ordered else 1.0
        out: list[SearchHit] = []
        for chunk_id, score in ordered:
            hit = records[chunk_id]
            hit.score = score / max_rrf
            out.append(hit)
        return out
