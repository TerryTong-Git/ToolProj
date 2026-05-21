from __future__ import annotations

import copy
import json
from typing import Any, cast


STAGE_SYSTEM_INSTRUCTION = {
    "nl": (
        "Return valid JSON matching the provided schema. "
        "Every required field must be present and non-empty. "
        "The simulation field must contain a brief but substantive explanation."
    ),
    "sim": (
        "Return valid JSON matching the provided schema. "
        "Every required field must be present and non-empty. "
        "The code field must contain complete executable Python for a function named solution(). "
        "The simulation field must contain a brief but substantive explanation of the code's behavior."
    ),
    "controlsim": (
        "Return valid JSON matching the provided schema. "
        "Every required field must be present and non-empty. "
        "The simulation field must contain a brief but substantive explanation."
    ),
}


def normalize_schema_for_structured_output(schema: dict[str, Any]) -> dict[str, Any]:
    def _normalize(node: Any) -> Any:
        if isinstance(node, list):
            return [_normalize(item) for item in node]
        if not isinstance(node, dict):
            return node

        normalized = copy.deepcopy(node)
        normalized.pop("default", None)
        normalized.pop("title", None)

        for key in ("properties", "$defs", "definitions"):
            if key in normalized and isinstance(normalized[key], dict):
                normalized[key] = {name: _normalize(value) for name, value in normalized[key].items()}

        for key in ("items", "additionalProperties", "contains", "not"):
            if key in normalized:
                normalized[key] = _normalize(normalized[key])

        for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
            if key in normalized and isinstance(normalized[key], list):
                normalized[key] = [_normalize(value) for value in normalized[key]]

        if isinstance(normalized.get("properties"), dict):
            prop_names = list(normalized["properties"].keys())
            normalized["required"] = prop_names
            normalized["additionalProperties"] = False

        if normalized.get("type") == "string" and "enum" not in normalized:
            normalized["minLength"] = max(1, int(normalized.get("minLength", 0)))

        return normalized

    return cast(dict[str, Any], _normalize(schema))


def structured_output_request(model_cls: Any, *, strict: bool = True) -> dict[str, Any]:
    schema = normalize_schema_for_structured_output(model_cls.model_json_schema())
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model_cls.__name__,
            "strict": strict,
            "schema": schema,
        },
    }


def validate_nonempty_fields(parsed: Any) -> str:
    payload = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
    for field_name, value in payload.items():
        if value is None:
            return f"empty_field:{field_name}"
        if isinstance(value, str) and not value.strip():
            return f"empty_field:{field_name}"
        if isinstance(value, (list, dict, tuple, set)) and not value:
            return f"empty_field:{field_name}"
    return "ok"


def response_format_key(response_format: dict[str, Any] | None) -> str:
    if response_format is None:
        return ""
    return json.dumps(response_format, sort_keys=True, ensure_ascii=False)
