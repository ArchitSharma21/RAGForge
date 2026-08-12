from ragforge.citations import normalize_citation_syntax, repair_missing_citations


def test_grouped_and_duplicate_trailing_citations_are_normalized():
    answer = "AI RMF covers four functions [D1, D2]. [D2] [D1]"
    assert normalize_citation_syntax(answer) == "AI RMF covers four functions [D1] [D2]."


def test_repair_skips_broad_preamble_ending_in_colon():
    answer = "The following documents contain escalation guidance:"
    sources = [
        {
            "id": "D1",
            "title": "acme_cloud_runbook.md",
            "snippet": "Incident escalation guidance and response procedures.",
        }
    ]
    repaired, count = repair_missing_citations(answer, sources)
    assert repaired == answer
    assert count == 0


def test_repair_adds_supported_citation_to_factual_bullet():
    answer = "* The Sev-1 acknowledgement target is five minutes."
    sources = [
        {
            "id": "D1",
            "title": "acme_cloud_runbook.md",
            "snippet": "The on-call engineer must acknowledge a Sev-1 page within five minutes.",
        }
    ]
    repaired, count = repair_missing_citations(answer, sources)
    assert "[D1]" in repaired
    assert count == 1


def test_repair_missing_citations_repairs_uncited_sentence_inside_cited_paragraph():
    from ragforge.citations import repair_missing_citations

    answer = (
        "Support tiers [T1]: Starter costs 0. "
        "Intermediate tiers include Team at 49 and Business at 199. "
        "Enterprise costs 799 [T1]."
    )
    sources = [{
        "id": "T1",
        "type": "table",
        "title": "support_matrix",
        "snippet": "Starter 0 Team 49 Business 199 Enterprise 799",
    }]
    repaired, count = repair_missing_citations(answer, sources, semantic_support=False)
    assert "Intermediate tiers include Team at 49 and Business at 199 [T1]." in repaired
    assert count >= 1
