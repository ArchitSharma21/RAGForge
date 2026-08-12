from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from .schemas import Chunk, Document, SourceProfile


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _sample_positions(n: int, count: int = 4) -> list[int]:
    if n <= 0:
        return []
    if n <= count:
        return list(range(n))
    # Deterministic coverage across the source: beginning, early-middle,
    # late-middle and end. This avoids a long PDF being represented only by
    # its first page without requiring an ingestion-time LLM call.
    raw = [0, round((n - 1) / 3), round(2 * (n - 1) / 3), n - 1]
    out: list[int] = []
    for idx in raw:
        if idx not in out:
            out.append(idx)
    return out[:count]


def build_source_profiles(documents: list[Document], chunks: list[Chunk]) -> dict[str, SourceProfile]:
    docs_by_source: dict[str, list[Document]] = defaultdict(list)
    chunks_by_source: dict[str, list[Chunk]] = defaultdict(list)
    for doc in documents:
        docs_by_source[doc.source].append(doc)
    for chunk in chunks:
        chunks_by_source[chunk.source].append(chunk)

    profiles: dict[str, SourceProfile] = {}
    for source in sorted(set(docs_by_source) | set(chunks_by_source)):
        docs = docs_by_source.get(source, [])
        source_chunks = chunks_by_source.get(source, [])
        safe_profile_chunks = [
            c for c in source_chunks if float(c.metadata.get("injection_score", 0.0)) < 0.5
        ] or source_chunks
        pages = {d.page for d in docs if d.page is not None}
        sections = {d.section for d in docs if d.section}

        representative_ids: list[str] = []
        excerpts: list[str] = []
        for idx in _sample_positions(len(safe_profile_chunks), 4):
            chunk = safe_profile_chunks[idx]
            representative_ids.append(chunk.id)
            excerpt = _clean(chunk.text)[:900]
            if excerpt and excerpt not in excerpts:
                excerpts.append(excerpt)

        suffix = Path(source).suffix.lower().lstrip(".") or "text"
        metadata_line = (
            f"Source: {source}. Type: {suffix}. Document units: {len(docs)}. "
            f"Chunks: {len(source_chunks)}. Pages: {len(pages)}. Sections: {len(sections)}."
        )
        profile_text = metadata_line
        if excerpts:
            profile_text += " Representative content: " + " | ".join(excerpts)

        profiles[source] = SourceProfile(
            source=source,
            file_type=suffix,
            document_units=len(docs),
            chunk_count=len(source_chunks),
            page_count=len(pages),
            section_count=len(sections),
            representative_chunk_ids=representative_ids,
            profile_text=profile_text[:5000],
        )
    return profiles


def corpus_manifest(
    profiles: dict[str, SourceProfile],
    tables: list[str] | None = None,
    max_chars: int = 9000,
    include_excerpts: bool = True,
) -> str:
    """Compact session-grounded description for the semantic planner.

    The manifest is deliberately extractive/deterministic: it costs no LLM call
    at ingestion, contains only corpus-derived text, and can be rebuilt whenever
    the workspace version changes.
    """
    if not profiles:
        table_text = ", ".join(tables or []) or "none"
        return f"No unstructured documents are indexed. Structured tables: {table_text}."

    lines = [f"Indexed sources: {len(profiles)}."]
    for i, profile in enumerate(profiles.values(), start=1):
        preview = profile.profile_text.split("Representative content:", 1)[-1].strip()
        preview = _clean(preview)[:500]
        line = (
            f"{i}. {profile.source} | type={profile.file_type} | units={profile.document_units} "
            f"| pages={profile.page_count} | chunks={profile.chunk_count}"
        )
        if include_excerpts:
            line += f" | excerpt={preview}"
        lines.append(line)
        if sum(len(x) for x in lines) >= max_chars:
            lines.append("(manifest truncated)")
            break
    if tables:
        lines.append("Structured tables: " + ", ".join(tables))
    else:
        lines.append("Structured tables: none")
    return "\n".join(lines)[:max_chars]


def profile_chunks(profiles: dict[str, SourceProfile]) -> list[Chunk]:
    """Represent each source as one synthetic *retrieval-only* chunk.

    These chunks are used to choose documents in the first stage of hierarchical
    retrieval. They are never surfaced as answer citations; generation always
    receives original document chunks.
    """
    out: list[Chunk] = []
    for i, profile in enumerate(profiles.values()):
        out.append(
            Chunk(
                id=f"source-profile-{i}",
                text=profile.profile_text,
                source=profile.source,
                metadata={"source_profile": True},
            )
        )
    return out
