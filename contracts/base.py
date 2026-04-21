"""Lightweight structured contracts for DeepGraph pipeline boundaries."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any, Iterable, Mapping, TypeVar


SCHEMA_VERSION = "v1"


class ContractValidationError(ValueError):
    """Raised when a structured contract is invalid."""


def load_jsonish(value: Any, default: Any):
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return default
    return default


def ensure_dict(value: Any) -> dict[str, Any]:
    loaded = load_jsonish(value, {})
    return loaded if isinstance(loaded, dict) else {}


def ensure_list(value: Any) -> list[Any]:
    loaded = load_jsonish(value, [])
    return loaded if isinstance(loaded, list) else []


def ensure_string_list(value: Any) -> list[str]:
    out: list[str] = []
    for item in ensure_list(value):
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def dump_contract_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {key: dump_contract_dict(val) for key, val in asdict(value).items()}
    if isinstance(value, list):
        return [dump_contract_dict(item) for item in value]
    if isinstance(value, dict):
        return {str(key): dump_contract_dict(val) for key, val in value.items()}
    return value


def coerce_optional_float(value: Any) -> float | None:
    if value in (None, "", []):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def coerce_optional_int(value: Any) -> int | None:
    if value in (None, "", []):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


T = TypeVar("T", bound="ContractRecord")


@dataclass
class ContractRecord:
    """Base dataclass for explicit pipeline hand-off objects."""

    schema_version: str = SCHEMA_VERSION

    @classmethod
    def contract_type(cls) -> str:
        return cls.__name__

    @classmethod
    def from_partial_dict(cls: type[T], payload: Mapping[str, Any]) -> T:
        known = {field.name for field in fields(cls)}
        kwargs = {name: payload[name] for name in known if name in payload}
        obj = cls(**kwargs)  # type: ignore[arg-type]
        obj.validate()
        return obj

    def validate(self) -> None:
        if not self.schema_version:
            raise ContractValidationError(f"{self.contract_type()} missing schema_version")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = dump_contract_dict(self)
        payload["contract_type"] = self.contract_type()
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, default=str)


def require_non_empty(name: str, value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        raise ContractValidationError(f"Missing required field: {name}")
    return text


def dedupe_strings(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out
