from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Protocol


def _validate_json(value: Any) -> None:
    if value is None or type(value) in (bool, int) or isinstance(value, str):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_json(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _validate_json(item)
        return
    raise ValueError("Only finite JSON values and string object keys are supported")


def canonical_json(value: Any) -> str:
    """Project encoding, not a claim of full RFC 8785 interoperability."""
    _validate_json(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class CheckpointSigner(Protocol):
    """Future signature interface; no signing implementation is claimed."""

    def sign(self, checkpoint_bytes: bytes) -> bytes: ...
    def verify(self, checkpoint_bytes: bytes, signature: bytes) -> bool: ...
