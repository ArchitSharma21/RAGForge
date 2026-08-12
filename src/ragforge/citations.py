from __future__ import annotations

import re
from typing import Any


def normalize_citation_syntax(answer: str) -> str:
    """Normalize grouped/redundant citation syntax without changing claims."""
    if not answer:
        return answer

    group_re = re.compile(r"\[((?:D|W|T)\d+(?:\s*,\s*(?:D|W|T)\d+)+)\]")
    citation_re = re.compile(r"\[((?:D|W|T)\d+)\]")

    def expand_group(match: re.Match[str]) -> str:
        ids = [part.strip() for part in match.group(1).split(",")]
        return " ".join(f"[{sid}]" for sid in ids)

    expanded = group_re.sub(expand_group, answer)
    cleaned_lines: list[str] = []
    for line in expanded.splitlines():
        # Only deduplicate a citation-only tail. This avoids stripping a
        # repeated citation that legitimately supports a second sentence.
        tail_match = re.search(r"((?:\s*\[(?:D|W|T)\d+\][\s.,;:]*)+)$", line)
        if not tail_match:
            cleaned_lines.append(line)
            continue
        tail = tail_match.group(1)
        ids = citation_re.findall(tail)
        if len(ids) <= 1:
            cleaned_lines.append(line)
            continue
        unique: list[str] = []
        for sid in ids:
            if sid not in unique:
                unique.append(sid)
        terminal = "." if "." in tail else ""
        prefix = line[: tail_match.start()].rstrip()
        normalized_tail = " ".join(f"[{sid}]" for sid in unique) + terminal
        cleaned_lines.append((prefix + " " + normalized_tail).strip())
    return "\n".join(cleaned_lines)


def repair_missing_citations(
    answer: str,
    sources: list[dict[str, Any]],
    *,
    semantic_support: bool = True,
) -> tuple[str, int]:
    """Attach citations only when an uncited factual unit clearly matches evidence.

    v1.7 repairs prose at sentence granularity. A paragraph that already has a
    citation in sentence one must not cause sentence two to be treated as cited.
    Bullets remain whole units so list formatting is preserved.
    """
    if not answer or not sources:
        return answer, 0
    answer = normalize_citation_syntax(answer)
    stop = {
        "the", "and", "for", "that", "with", "from", "this", "are", "was", "were", "has", "have",
        "into", "about", "their", "they", "its", "which", "what", "when", "where", "than", "then",
        "also", "using", "used", "user", "users", "document", "documents", "source", "sources",
    }

    def toks(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.%-]{2,}", (text or "").lower())
            if token not in stop
        }

    evidence: list[tuple[str, set[str]]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for source in sources:
        sid = str(source.get("id", ""))
        if not re.fullmatch(r"(?:D|W|T)\d+", sid):
            continue
        text = f"{source.get('title', '')} {source.get('snippet', '')}"
        evidence.append((sid, toks(text)))
        by_id[sid] = source
    if not evidence:
        return answer, 0

    semantic_vectors = None
    semantic_ids: list[str] = []
    if semantic_support:
        try:
            import numpy as np
            from .retrieval import ModelRegistry

            semantic_ids = [sid for sid, _ in evidence]
            texts = [
                f"{by_id[sid].get('title', '')} {by_id[sid].get('snippet', '')}"[:2400]
                for sid in semantic_ids
            ]
            semantic_vectors = np.asarray(list(ModelRegistry.embedding().passage_embed(texts)), dtype=float)
            norms = np.linalg.norm(semantic_vectors, axis=1, keepdims=True) + 1e-9
            semantic_vectors = semantic_vectors / norms
        except Exception:
            semantic_vectors = None

    def choose_ids(plain: str) -> list[str]:
        unit_tokens = toks(plain)
        if not unit_tokens:
            return []
        ranked: list[tuple[int, float, str]] = []
        for sid, source_tokens in evidence:
            overlap = len(unit_tokens & source_tokens)
            score = overlap / max(1, min(len(unit_tokens), 10))
            ranked.append((overlap, score, sid))
        ranked.sort(reverse=True)
        best_overlap, best_score, best_sid = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        selected_ids: list[str] = []
        if best_overlap >= 2 and (best_score >= 0.20 or best_score >= second_score + 0.10):
            selected_ids = [best_sid]
            if len(ranked) > 1:
                second_overlap, second_support, second_sid = ranked[1]
                if (
                    second_overlap >= 2
                    and second_support >= 0.20
                    and second_support >= best_score * 0.65
                ):
                    selected_ids.append(second_sid)
        elif semantic_vectors is not None and semantic_ids:
            try:
                import numpy as np
                from .retrieval import ModelRegistry

                vec = np.asarray(list(ModelRegistry.embedding().query_embed([plain]))[0], dtype=float)
                vec = vec / (np.linalg.norm(vec) + 1e-9)
                sims = semantic_vectors @ vec
                order = np.argsort(sims)[::-1]
                best_idx = int(order[0])
                best_sem = float(sims[best_idx])
                second_sem = float(sims[int(order[1])]) if len(order) > 1 else -1.0
                if best_sem >= 0.68 and (best_sem - second_sem >= 0.055 or best_sem >= 0.78):
                    selected_ids = [semantic_ids[best_idx]]
            except Exception:
                selected_ids = []
        return selected_ids

    def repair_unit(unit: str) -> tuple[str, int]:
        stripped = unit.strip()
        plain = re.sub(r"[`*_#>-]", "", stripped).strip()
        if (
            not stripped
            or re.search(r"\[(?:D|W|T)\d+\]", unit)
            or stripped.startswith("```")
            or stripped.endswith(":")
            or len(plain) < 24
        ):
            return unit, 0
        selected_ids = choose_ids(plain)
        if not selected_ids:
            return unit, 0
        citation_text = " ".join(f"[{sid}]" for sid in selected_ids)
        trimmed = unit.rstrip()
        terminal = trimmed[-1] if trimmed and trimmed[-1] in ".!?" else ""
        if terminal:
            trimmed = trimmed[:-1].rstrip()
            return f"{trimmed} {citation_text}{terminal}", len(selected_ids)
        return f"{trimmed} {citation_text}", len(selected_ids)

    repaired = 0
    out: list[str] = []
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("```") or stripped.endswith(":"):
            out.append(line)
            continue

        # Keep list items intact. The evaluator also treats one list item as one
        # factual unit, so this preserves readable Markdown and avoids citation
        # decoration on every short clause inside a bullet.
        if re.match(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", line):
            repaired_line, count = repair_unit(line)
            out.append(repaired_line)
            repaired += count
            continue

        # Move a citation written after sentence punctuation back onto that
        # sentence, then protect common abbreviations before splitting.
        split_text = re.sub(
            r"([.!?])\s+((?:\[(?:D|W|T)\d+(?:\s*,\s*(?:D|W|T)\d+)*\]\s*)+)",
            r" \2\1 ",
            line,
        )
        protected = (
            split_text.replace("vs.", "vs<prd>")
            .replace("e.g.", "e<prd>g<prd>")
            .replace("i.e.", "i<prd>e<prd>")
            .replace("etc.", "etc<prd>")
        )
        units = [part.strip().replace("<prd>", ".") for part in re.split(r"(?<=[.!?])\s+", protected)]
        repaired_units: list[str] = []
        for unit in units:
            if not unit:
                continue
            repaired_unit, count = repair_unit(unit)
            repaired_units.append(repaired_unit)
            repaired += count
        out.append(" ".join(repaired_units) if repaired_units else line)

    return normalize_citation_syntax("\n".join(out)), repaired

