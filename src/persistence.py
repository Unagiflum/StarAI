"""Shared JSON persistence primitives for user-editable configuration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import os
from pathlib import Path
import tempfile
from typing import Any, TypeVar


class PersistenceValidationError(ValueError):
    """Raised when JSON has the wrong shape for a persisted domain value."""


EXPECTED_READ_ERRORS = (
    FileNotFoundError,
    json.JSONDecodeError,
    UnicodeDecodeError,
    OSError,
    PersistenceValidationError,
)


def read_json(path: Path) -> Any:
    """Read one UTF-8 JSON document.

    I/O, decoding, and JSON errors deliberately propagate so a domain repository
    can apply its own compatibility fallback.
    """
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def atomic_write_json(path: Path, value: Any) -> None:
    """Write UTF-8 JSON without exposing a partially written target file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(value, file, indent=4)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass


T = TypeVar("T")


def require_object(value: Any, domain: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PersistenceValidationError(f"{domain} must be a JSON object")
    return value


def merge_validated_defaults(
    defaults: Mapping[str, T],
    loaded: Mapping[str, Any],
    validator: Callable[[str, Any], T],
) -> dict[str, T]:
    """Merge known fields, falling back per field when a value is invalid."""
    merged = dict(defaults)
    for key in defaults:
        if key not in loaded:
            continue
        try:
            merged[key] = validator(key, loaded[key])
        except PersistenceValidationError:
            pass
    return merged
