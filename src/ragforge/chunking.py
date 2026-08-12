from __future__ import annotations

import re
import uuid

import numpy as np
from .config import get_settings
from .schemas import Chunk, Document
from .security import prompt_injection_score


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)


def chunk_documents(documents: list[Document], semantic: bool = False) -> list[Chunk]:
    settings = get_settings()
    chunks: list[Chunk] = []
    for doc in documents:
        sentences = _split_sentences(doc.text)
        if semantic and len(sentences) >= 4:
            sentences = _semantic_groups(sentences)
        if not sentences:
            continue
        current: list[str] = []
        current_len = 0
        for sentence in sentences:
            if current and current_len + len(sentence) + 1 > settings.chunk_size_chars:
                text = " ".join(current).strip()
                chunks.append(_make_chunk(doc, text))
                overlap: list[str] = []
                overlap_len = 0
                for item in reversed(current):
                    if overlap_len + len(item) > settings.chunk_overlap_chars:
                        break
                    overlap.insert(0, item)
                    overlap_len += len(item) + 1
                current = overlap
                current_len = sum(len(x) + 1 for x in current)
            current.append(sentence)
            current_len += len(sentence) + 1
        if current:
            chunks.append(_make_chunk(doc, " ".join(current).strip()))
    return chunks[: settings.max_chunks_per_session]



def _semantic_groups(sentences: list[str]) -> list[str]:
    """Group adjacent sentences at semantic breakpoints before size-based chunking."""
    try:
        from .retrieval import ModelRegistry
        vectors = np.asarray(list(ModelRegistry.embedding().passage_embed(sentences)))
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
        vectors = vectors / norms
        sims = np.sum(vectors[:-1] * vectors[1:], axis=1)
        threshold = float(np.percentile(sims, 20))
        groups: list[str] = []
        current = [sentences[0]]
        current_len = len(sentences[0])
        for i, sentence in enumerate(sentences[1:]):
            should_break = sims[i] <= threshold and current_len >= 500
            if should_break:
                groups.append(" ".join(current))
                current = [sentence]
                current_len = len(sentence)
            else:
                current.append(sentence)
                current_len += len(sentence) + 1
        if current:
            groups.append(" ".join(current))
        return groups
    except Exception:
        return sentences


def _make_chunk(doc: Document, text: str) -> Chunk:
    metadata = dict(doc.metadata)
    metadata["injection_score"] = prompt_injection_score(text)
    return Chunk(
        id=str(uuid.uuid4()),
        text=text,
        source=doc.source,
        page=doc.page,
        section=doc.section,
        metadata=metadata,
    )
