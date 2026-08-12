from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .schemas import QueryPlan, SearchHit

_STOPWORDS = {
    "what", "which", "when", "where", "who", "why", "how", "does", "do", "did", "is", "are", "was", "were",
    "the", "a", "an", "and", "or", "of", "to", "for", "from", "in", "on", "with", "about", "our", "this",
    "that", "these", "those", "current", "indexed", "document", "documents", "collection", "please", "give",
}


@dataclass(slots=True)
class EvidenceCompressionDecision:
    texts: dict[str, str]
    used: bool
    reason: str
    chunks: int
    chars_before: int
    chars_after: int
    tokens_est_before: int
    tokens_est_after: int

    @property
    def reduction_ratio(self) -> float:
        if self.chars_before <= 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - (self.chars_after / self.chars_before)))

    def trace_fields(self) -> dict[str, object]:
        return {
            "evidence_compression_used": self.used,
            "evidence_compression_reason": self.reason,
            "evidence_compression_chunks": self.chunks,
            "evidence_chars_before_compression": self.chars_before,
            "evidence_chars_after_compression": self.chars_after,
            "evidence_tokens_est_before_compression": self.tokens_est_before,
            "evidence_tokens_est_after_compression": self.tokens_est_after,
            "evidence_compression_reduction_pct": round(self.reduction_ratio * 100.0, 1),
        }


def _tokens_est(chars: int) -> int:
    return int(math.ceil(max(0, chars) / 4.0))


def _query_terms(query: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_]+", query or "")
        if len(token) >= 3 and token.lower() not in _STOPWORDS
    }


def _split_units(text: str) -> list[str]:
    cleaned = re.sub(r"\r\n?", "\n", text or "")
    # Keep bullet/list lines as their own candidate units while splitting long
    # prose paragraphs into sentences. This is intentionally lightweight and
    # deterministic so compression never spends an LLM request.
    units: list[str] = []
    for line in cleaned.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if re.match(r"^(?:[-*+]\s+|\d+[.)]\s+)", line):
            units.append(line)
            continue
        pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9#*`])", line)
        units.extend(piece.strip() for piece in pieces if piece.strip())
    return units or [re.sub(r"\s+", " ", cleaned).strip()]


def _entity_labels(text: str) -> list[str]:
    """Return compact entity labels such as ``sev-1`` or ``tier-2``.

    These labels often disambiguate nearby sibling facts that otherwise share
    the same generic wording (for example, ``Sev-1`` and ``Sev-2`` both having
    an acknowledgement target).
    """
    labels = []
    for prefix, number in re.findall(r"\b([A-Za-z][A-Za-z0-9_]*)-(\d+)\b", text or ""):
        labels.append(f"{prefix.lower()}-{number}")
    return labels


def _entity_family(label: str) -> str:
    return label.rsplit("-", 1)[0] if "-" in label else label


def _score_unit(
    unit: str,
    terms: set[str],
    query_numbers: set[str],
    *,
    query_entities: set[str] | None = None,
    context_entities: set[str] | None = None,
) -> float:
    lower = unit.lower()
    unit_terms = set(re.findall(r"[A-Za-z0-9_]+", lower))
    overlap = len(terms & unit_terms)
    number_hits = len(query_numbers & unit_terms)
    score = overlap * 3.0 + number_hits * 2.0 + min(1.5, len(unit) / 500.0)

    query_entities = query_entities or set()
    context_entities = context_entities or set(_entity_labels(unit))
    if query_entities:
        # Strongly prefer evidence bound to the exact entity named by the query.
        # If a nearby generic sentence inherits a sibling entity (e.g. a
        # ``15 minutes`` sentence immediately after ``Sev-2``), penalize it so
        # it cannot outrank the explicitly bound ``Sev-1`` fact.
        exact = query_entities & context_entities
        if exact:
            score += 8.0 * len(exact)
        else:
            query_families = {_entity_family(label) for label in query_entities}
            sibling = {
                label for label in context_entities
                if _entity_family(label) in query_families and label not in query_entities
            }
            if sibling:
                score -= 8.0 * len(sibling)
    return score


def compress_text_for_query(query: str, text: str, *, max_chars: int = 900, max_units: int = 3) -> str:
    original = re.sub(r"\s+", " ", text or "").strip()
    if len(original) <= max_chars:
        return original

    units = _split_units(text)
    terms = _query_terms(query)
    query_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", query or ""))
    query_entities = set(_entity_labels(query))

    # Short follow-up sentences often omit the entity they belong to. Preserve
    # that local binding for one sentence so sibling facts remain distinct.
    # Example: ``A Sev-2 incident ... . The acknowledgement target is 15
    # minutes.`` The second sentence should inherit ``sev-2``.
    unit_context_entities: list[set[str]] = []
    previous_entities: set[str] = set()
    for unit in units:
        explicit = set(_entity_labels(unit))
        if explicit:
            current = explicit
            previous_entities = explicit
        else:
            current = set(previous_entities)
            previous_entities = set()
        unit_context_entities.append(current)

    ranked = sorted(
        enumerate(units),
        key=lambda item: (
            _score_unit(
                item[1],
                terms,
                query_numbers,
                query_entities=query_entities,
                context_entities=unit_context_entities[item[0]],
            ),
            -item[0],
        ),
        reverse=True,
    )
    chosen_idx = sorted(idx for idx, _ in ranked[: max(1, max_units)])
    chosen = [units[idx] for idx in chosen_idx]
    candidate = " ".join(chosen).strip()
    if not candidate:
        candidate = original[:max_chars]
    if len(candidate) > max_chars:
        candidate = candidate[: max_chars - 3].rsplit(" ", 1)[0].rstrip() + "..."
    return candidate


def focused_evidence_compression(
    hits: list[SearchHit],
    plan: QueryPlan,
    *,
    query: str,
    enabled: bool = True,
) -> EvidenceCompressionDecision:
    before_chars = sum(len(hit.chunk.text or "") for hit in hits)

    def decision(texts: dict[str, str], used: bool, reason: str) -> EvidenceCompressionDecision:
        after_chars = sum(len(texts.get(hit.chunk.id, hit.chunk.text or "")) for hit in hits)
        return EvidenceCompressionDecision(
            texts=texts,
            used=used,
            reason=reason,
            chunks=len(hits),
            chars_before=before_chars,
            chars_after=after_chars,
            tokens_est_before=_tokens_est(before_chars),
            tokens_est_after=_tokens_est(after_chars),
        )

    original = {hit.chunk.id: hit.chunk.text or "" for hit in hits}
    if not hits:
        return decision(original, False, "no_document_evidence")
    if not enabled:
        return decision(original, False, "disabled_by_user")
    if plan.task_type not in {"fact_lookup", "followup"}:
        return decision(original, False, "broad_or_multi_source_task")
    if plan.retrieval_strategy not in {"semantic", "hierarchical"}:
        return decision(original, False, "strategy_requires_full_chunks")

    compressed = {
        hit.chunk.id: compress_text_for_query(query, hit.chunk.text or "")
        for hit in hits
    }
    after_chars = sum(len(text) for text in compressed.values())
    if after_chars >= before_chars * 0.92:
        return decision(original, False, "minimal_compression_gain")
    return decision(compressed, True, "focused_sentence_selection")
