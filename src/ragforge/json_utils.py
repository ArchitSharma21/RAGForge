from __future__ import annotations

import ast
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any


def to_jsonable(value: Any) -> Any:
    """Return plain JSON-compatible Python values.

    Gradio and Pydantic can wrap JSON payloads in RootModel-like objects. Saved
    evaluation reports should have exactly the same shape whether they are fresh,
    restored from disk, returned through FastAPI, or rendered in the UI.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, str) and value.startswith("root="):
            try:
                return to_jsonable(ast.literal_eval(value[5:]))
            except Exception:
                return value
        return value

    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(mode="json")
        except TypeError:
            dumped = value.model_dump()
        return to_jsonable(dumped)

    if is_dataclass(value):
        return to_jsonable(asdict(value))

    if isinstance(value, Mapping):
        plain = {str(key): to_jsonable(item) for key, item in value.items()}
        if set(plain) == {"root"} and isinstance(plain["root"], (dict, list)):
            return plain["root"]
        return plain

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_jsonable(item) for item in value]

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return to_jsonable(item())
        except Exception:
            pass
    return str(value)


def pretty_json(value: Any) -> str:
    return json.dumps(to_jsonable(value), indent=2, ensure_ascii=False)
