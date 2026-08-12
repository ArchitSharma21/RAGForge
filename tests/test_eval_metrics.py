from ragforge.eval_metrics import answer_key_match, citation_metrics, percentile, scalar_value_match, source_metrics


def test_source_metrics_reward_early_relevant_source():
    metrics = source_metrics(["wrong.md", "right.md", "other.md"], ["right.md"])
    assert metrics["source_recall@5"] == 1.0
    assert metrics["source_mrr"] == 0.5
    assert metrics["source_hit@1"] == 0.0
    assert 0 < metrics["source_precision@5"] < 1
    assert 0 <= metrics["source_ap@5"] <= 1
    assert 0 <= metrics["source_ndcg@5"] <= 1


def test_source_metrics_deduplicate_repeated_chunks_and_bound_ap():
    metrics = source_metrics(["nist.pdf"] * 5, ["nist.pdf"])
    assert metrics["source_hit@1"] == 1.0
    assert metrics["source_mrr"] == 1.0
    assert metrics["source_ap@5"] == 1.0
    assert metrics["source_ndcg@5"] == 1.0
    assert metrics["source_duplicate_rate@5"] == 0.8


def test_citation_metrics_distinguish_validity_and_coverage():
    answer = "The target is five minutes [D1]. A second unsupported statement is also present."
    sources = [{"id": "D1", "type": "document", "title": "runbook.md"}]
    metrics = citation_metrics(answer, sources)
    assert metrics["citation_validity"] == 1.0
    assert 0 < metrics["citation_coverage"] < 1.0


def test_citation_metrics_reject_unknown_ids():
    metrics = citation_metrics("Claim [D9].", [{"id": "D1"}])
    assert metrics["citation_validity"] == 0.0


def test_answer_key_supports_all_and_any():
    assert answer_key_match("Govern, Map, Measure, Manage", {"expected_all": ["govern", "map", "measure", "manage"]})
    assert answer_key_match("Enterprise is fastest", {"expected_any": ["enterprise", "business"]})
    assert not answer_key_match("Team is fastest", {"expected_any": ["enterprise"]})
    # Numeric answer labels must not match inside a different number. This caught
    # a real regression where "15 minutes" passed a "5 min" answer key.
    assert answer_key_match("The target is 5 minutes.", {"expected_any": ["5 minutes", "5 min"]})
    assert not answer_key_match("The target is 15 minutes.", {"expected_any": ["5 minutes", "5 min"]})
    # A later correction must not rescue an explicitly wrong primary answer.
    contradictory = (
        "The Sev-1 acknowledgement target is 15 minutes [D1]. "
        "(Note: The runbook also specifies that the on-call engineer must "
        "acknowledge a Sev-1 page within 5 minutes [D1].)"
    )
    assert not answer_key_match(
        contradictory,
        {
            "question": "What is the Sev-1 acknowledgement target?",
            "expected_any": ["5 minutes", "5 min"],
        },
    )
    assert answer_key_match(
        "The Sev-1 acknowledgement target is 5 minutes [D1]. "
        "A Sev-2 acknowledgement target is 15 minutes [D1].",
        {
            "question": "What is the Sev-1 acknowledgement target?",
            "expected_any": ["5 minutes", "5 min"],
        },
    )


def test_percentile_interpolates():
    assert percentile([100, 200, 300], 0.5) == 200
    assert percentile([], 0.95) == 0.0


def test_scalar_value_match_handles_boolean_numeric_and_text_values():
    assert scalar_value_match(True, True)
    assert scalar_value_match("true", True)
    assert scalar_value_match(199, 199)
    assert scalar_value_match(199.0, 199)
    assert scalar_value_match("Enterprise", "enterprise")
    assert not scalar_value_match(False, True)


def test_grouped_citation_ids_are_supported():
    from ragforge.eval_metrics import citation_metrics

    sources = [{"id": "D1"}, {"id": "D2"}]
    metrics = citation_metrics(
        "The system uses hybrid retrieval with two supporting sources [D1, D2].",
        sources,
    )
    assert metrics["citation_count"] == 2
    assert metrics["citation_validity"] == 1.0
    assert metrics["citation_coverage"] == 1.0
