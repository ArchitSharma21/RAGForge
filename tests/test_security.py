from pathlib import Path
import zipfile
import pytest

from ragforge.security import safe_extract_zip, validate_readonly_sql


def test_readonly_sql_accepts_select_and_adds_limit():
    sql = validate_readonly_sql("SELECT * FROM support_matrix")
    assert sql.lower().endswith("limit 200")


def test_readonly_sql_rejects_mutation():
    with pytest.raises(ValueError):
        validate_readonly_sql("DROP TABLE support_matrix")


def test_zip_slip_is_rejected(tmp_path: Path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "nope")
    with pytest.raises(ValueError):
        safe_extract_zip(archive, tmp_path / "out")
