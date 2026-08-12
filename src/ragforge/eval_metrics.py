from __future__ import annotations

import math
import re
import statistics
from typing import Any

_CITATION_RE = re.compile(r"\[((?:D|W|T)\d+)\]")
_CITATION_GROUP_RE = re.compile(r"\[((?:D|W|T)\d+(?:\s*,\s*(?:D|W|T)\d+)*)\]")


def extract_citation_ids(text: str) -> list[str]:
    ids: list[str] = []
    for match in _CITATION_GROUP_RE.findall(text or ""):
        for part in match.split(","):
            sid = part.strip()
            if sid:
                ids.append(sid)
    return ids


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _contains_expected_term(answer: str, term: str) -> bool:
    """Match a labeled answer term without accepting alphanumeric substrings.

    Plain ``term in answer`` makes numeric labels unsafe: for example ``5 min``
    is a substring of ``15 minutes``. Benchmark matching should recognize the
    expected phrase as its own token/phrase while remaining tolerant of ordinary
    whitespace differences.
    """
    text = re.sub(r"\s+", " ", (answer or "").casefold()).strip()
    expected = re.sub(r"\s+", " ", (term or "").casefold()).strip()
    if not expected:
        return False
    pattern = re.escape(expected).replace(r"\ ", r"\s+")
    if expected[0].isalnum():
        pattern = r"(?<![0-9A-Za-z])" + pattern
    if expected[-1].isalnum():
        pattern = pattern + r"(?![0-9A-Za-z])"
    return re.search(pattern, text) is not None




_NUMERIC_TOKEN_RE = re.compile(r"(?<![0-9A-Za-z])\d+(?:\.\d+)?%?(?![0-9A-Za-z])")
_QA_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "does", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "the", "to", "what", "which",
    "with", "within", "do", "has", "have", "that", "this",
}


def _numeric_tokens(text: str) -> set[str]:
    return {m.group(0).casefold() for m in _NUMERIC_TOKEN_RE.finditer(text or "")}


def _content_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]*", (text or "").casefold())
    return {tok for tok in tokens if tok not in _QA_STOPWORDS and len(tok) > 1}


def _has_primary_numeric_conflict(answer: str, case: dict[str, Any]) -> bool:
    """Reject a wrong primary numeric answer that is later hedged with the right value.

    A phrase-only matcher can still be fooled by a response such as
    ``"The target is 15 minutes. Note: the source says 5 minutes."``.  For
    benchmark cases whose labeled answer contains a number, identify the sentence
    most directly about the question.  If that primary sentence asserts a
    different numeric value and contains none of the labeled numeric values, the
    response is contradictory and should not pass merely because a later sentence
    mentions the expected value.

    Numbers already present in the question (for example the ``1`` in ``Sev-1``)
    are ignored so identifiers are not mistaken for answer values.
    """
    expected_terms = [str(x) for x in case.get("expected_all", [])] + [str(x) for x in case.get("expected_any", [])]
    expected_numbers: set[str] = set()
    for term in expected_terms:
        expected_numbers.update(_numeric_tokens(term))
    if not expected_numbers:
        return False

    question = str(case.get("question", ""))
    question_numbers = _numeric_tokens(question)
    question_tokens = _content_tokens(question)
    if not question_tokens:
        return False

    # Split prose into sentence-like units while also respecting bullet/newline boundaries.
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", answer or "") if part.strip()]
    candidates: list[tuple[int, int, str, set[str]]] = []
    for idx, sentence in enumerate(sentences):
        nums = _numeric_tokens(sentence) - question_numbers
        if not nums:
            continue
        overlap = len(question_tokens & _content_tokens(sentence))
        if overlap <= 0:
            continue
        candidates.append((overlap, -idx, sentence, nums))

    if not candidates:
        return False

    # Highest question-token overlap wins; on a tie prefer the earlier statement.
    _, _, _primary, primary_numbers = max(candidates, key=lambda row: (row[0], row[1]))
    return bool(primary_numbers and primary_numbers.isdisjoint(expected_numbers))

def answer_key_match(answer: str, case: dict[str, Any]) -> bool:
    if _has_primary_numeric_conflict(answer, case):
        return False
    expected_all = [str(x) for x in case.get("expected_all", [])]
    expected_any = [str(x) for x in case.get("expected_any", [])]
    if expected_all and not all(_contains_expected_term(answer, term) for term in expected_all):
        return False
    if expected_any and not any(_contains_expected_term(answer, term) for term in expected_any):
        return False
    return bool(expected_all or expected_any)



def missing_answer_match(answer: str, case: dict[str, Any] | None = None) -> bool:
    """Recognize a grounded "not present in the evidence" answer.

    Missing-answer evaluation should reward calibrated uncertainty, not require a
    single canned phrase. The matcher therefore accepts benchmark-specific cues
    plus a conservative generic vocabulary for absence/insufficiency.
    """
    text = re.sub(r"\s+", " ", (answer or "").strip().casefold())
    if not text:
        return False
    case = case or {}
    expected = [str(x).casefold() for x in case.get("expected_missing_any", [])]
    generic = [
        "not specified", "does not specify", "doesn't specify",
        "not provided", "does not provide", "doesn't provide",
        "does not mention", "doesn't mention", "do not mention", "not mentioned",
        "does not contain", "doesn't contain", "no information", "no fee information",
        "insufficient to answer", "insufficient evidence", "cannot determine", "can't determine",
        "not stated", "not available in", "not present in", "no evidence of",
    ]
    return any(cue in text for cue in [*expected, *generic] if cue)


def substantive_claim_units(answer: str) -> list[str]:
    """Extract Markdown-aware factual units for citation coverage.

    Headings and generic list introductions are presentation structure, not
    factual claims. Short numbered/bulleted values are factual units even when
    they are much shorter than prose sentences.
    """
    units: list[str] = []
    in_code = False
    for raw_line in (answer or "").splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not line:
            continue
        if re.match(r"^#{1,6}\s+", line):
            continue

        is_list = bool(re.match(r"^(?:[-*+]\s+|\d+[.)]\s+)", line))
        content = re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", line).strip()
        plain = re.sub(r"\[(?:D|W|T)\d+(?:\s*,\s*(?:D|W|T)\d+)*\]", "", content)
        plain = re.sub(r"[`*_#>]", "", plain).strip()
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_.%$+-]*", plain)

        # Preambles such as "The following documents contain:" introduce the
        # claims in following bullets and should not depress citation coverage.
        if not is_list and plain.endswith(":"):
            continue

        if is_list:
            if len(words) >= 1 and len(plain) >= 3:
                units.append(content)
            continue

        # Split long prose lines into sentence-level claims. Citations are often
        # written after punctuation (``claim. [D1]``); move that citation tail
        # onto the claim before splitting. Protect common abbreviations such as
        # ``vs.`` so they do not become fake uncited sentence fragments.
        split_text = re.sub(
            r"([.!?])\s+((?:\[(?:D|W|T)\d+(?:\s*,\s*(?:D|W|T)\d+)*\]\s*)+)",
            r" \2\1 ",
            content,
        )
        protected = (
            split_text.replace("vs.", "vs<prd>")
            .replace("e.g.", "e<prd>g<prd>")
            .replace("i.e.", "i<prd>e<prd>")
            .replace("etc.", "etc<prd>")
        )
        segments = [
            seg.strip().replace("<prd>", ".")
            for seg in re.split(r"(?<=[.!?])\s+", protected)
            if seg.strip()
        ]
        for segment in segments:
            segment_plain = re.sub(r"\[(?:D|W|T)\d+(?:\s*,\s*(?:D|W|T)\d+)*\]", "", segment)
            segment_plain = re.sub(r"[`*_#>]", "", segment_plain).strip()
            seg_words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_.%$+-]*", segment_plain)
            if len(segment_plain) >= 24 or len(seg_words) >= 5:
                units.append(segment)
    return units

def scalar_value_match(observed: Any, expected: Any) -> bool:
    """Compare tabular scalar values without relying on Markdown rendering.

    DuckDB/pandas may expose booleans and numerics as numpy scalar types. The
    benchmark should judge the computed value itself, not whether a rendered
    table happened to spell a boolean as ``true``, ``True`` or ``1``.
    """
    try:
        if hasattr(observed, "item"):
            observed = observed.item()
    except Exception:
        pass

    if isinstance(expected, bool):
        if isinstance(observed, bool):
            return observed is expected
        text = str(observed).strip().lower()
        truthy = {"true", "1", "yes", "y", "t"}
        falsy = {"false", "0", "no", "n", "f"}
        return text in (truthy if expected else falsy)

    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return abs(float(observed) - float(expected)) <= 1e-9
        except Exception:
            return False

    return str(observed).strip().casefold() == str(expected).strip().casefold()


def _unique_sources(values: list[str], k: int = 5) -> list[str]:
    """Return the first k distinct sources while preserving retrieval order.

    RAG retrieval commonly returns several chunks from the same file. Source-level
    metrics must not count the same relevant file multiple times, otherwise AP can
    exceed 1.0 and source precision becomes difficult to interpret.
    """
    out: list[str] = []
    seen: set[str] = set()
    for value in values[:k]:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def source_metrics(returned: list[str], relevant: list[str]) -> dict[str, float]:
    relevant_set = set(relevant)
    if not relevant_set:
        return {
            "source_precision@5": 1.0,
            "source_recall@5": 1.0,
            "source_hit@1": 1.0,
            "source_mrr": 1.0,
            "source_ap@5": 1.0,
            "source_ndcg@5": 1.0,
            "source_duplicate_rate@5": 0.0,
        }

    raw_top = returned[:5]
    ranked = _unique_sources(returned, 5)
    hits = [1 if source in relevant_set else 0 for source in ranked]

    precision = safe_div(sum(hits), len(ranked))
    recall = safe_div(len(set(ranked) & relevant_set), len(relevant_set))
    hit_at_1 = float(bool(ranked and ranked[0] in relevant_set))

    reciprocal_rank = 0.0
    precisions_at_relevant: list[float] = []
    relevant_seen = 0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            relevant_seen += 1
            if reciprocal_rank == 0.0:
                reciprocal_rank = 1.0 / rank
            precisions_at_relevant.append(relevant_seen / rank)

    # Average Precision divides by the number of relevant sources that could be
    # retrieved within the cutoff, and each source contributes at most once.
    ap = safe_div(sum(precisions_at_relevant), min(len(relevant_set), 5))
    ap = max(0.0, min(1.0, ap))

    dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, start=1))
    ideal_hits = min(len(relevant_set), 5)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = safe_div(dcg, idcg)

    duplicate_rate = safe_div(len(raw_top) - len(set(raw_top)), len(raw_top)) if raw_top else 0.0

    return {
        "source_precision@5": max(0.0, min(1.0, precision)),
        "source_recall@5": max(0.0, min(1.0, recall)),
        "source_hit@1": hit_at_1,
        "source_mrr": max(0.0, min(1.0, reciprocal_rank)),
        "source_ap@5": ap,
        "source_ndcg@5": max(0.0, min(1.0, ndcg)),
        "source_duplicate_rate@5": max(0.0, min(1.0, duplicate_rate)),
    }


def citation_metrics(answer: str, result_sources: list[dict[str, Any]]) -> dict[str, float | int]:
    cited = extract_citation_ids(answer or "")
    valid_ids = {str(source.get("id", "")) for source in result_sources}
    valid = sum(1 for citation in cited if citation in valid_ids)
    validity = safe_div(valid, len(cited)) if cited else 0.0

    units = substantive_claim_units(answer or "")
    cited_units = sum(1 for unit in units if extract_citation_ids(unit))
    coverage = safe_div(cited_units, len(units)) if units else 0.0
    return {
        "citation_count": len(cited),
        "citation_validity": validity,
        "citation_coverage": coverage,
        "substantive_units": len(units),
        "cited_units": cited_units,
    }
