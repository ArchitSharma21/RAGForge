from ragforge.json_utils import pretty_json, to_jsonable


def test_json_utils_unwrap_root_mapping():
    value = {"root": {"summary": {"quality_grade": "A"}}}
    assert to_jsonable(value) == {"summary": {"quality_grade": "A"}}
    rendered = pretty_json(value)
    assert '"summary"' in rendered
    assert "root=" not in rendered


def test_json_utils_recovers_legacy_root_repr_string():
    value = "root={'summary': {'quality_grade': 'A'}, 'ok': True}"
    normalized = to_jsonable(value)
    assert normalized["summary"]["quality_grade"] == "A"
    assert normalized["ok"] is True
