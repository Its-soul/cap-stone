from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .crypto import canonical_json, sha256
from .models import utc_now

GENESIS = "0" * 64


@dataclass(frozen=True)
class Checkpoint:
    length: int
    head_hash: str


@dataclass(frozen=True)
class AuditEvent:
    sequence: int
    kind: str
    timestamp: str
    payload_json: str
    previous_hash: str
    event_hash: str

    def body(self) -> dict:
        return {"sequence": self.sequence, "kind": self.kind, "timestamp": self.timestamp,
                "payload": json.loads(self.payload_json), "previous_hash": self.previous_hash}

    def to_dict(self) -> dict:
        return {**self.body(), "event_hash": self.event_hash}


class AuditJournal:
    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    def append(self, kind: str, payload: Any) -> AuditEvent:
        body = {"sequence": len(self._events), "kind": kind, "timestamp": utc_now(),
                "payload": payload, "previous_hash": self.checkpoint().head_hash}
        event = AuditEvent(body["sequence"], kind, body["timestamp"], canonical_json(payload),
                           body["previous_hash"], sha256(body))
        self._events.append(event)
        return event

    def checkpoint(self) -> Checkpoint:
        return Checkpoint(len(self._events), self._events[-1].event_hash if self._events else GENESIS)

    def export(self) -> list[dict]:
        return [event.to_dict() for event in self._events]

    @staticmethod
    def verify(records: list[dict], expected: Checkpoint | None = None) -> bool:
        previous = GENESIS
        try:
            for sequence, record in enumerate(records):
                if set(record) != {"sequence", "kind", "timestamp", "payload", "previous_hash", "event_hash"}:
                    return False
                body = {key: value for key, value in record.items() if key != "event_hash"}
                if (type(record["sequence"]) is not int or record["sequence"] != sequence
                        or record["previous_hash"] != previous or sha256(body) != record["event_hash"]):
                    return False
                previous = record["event_hash"]
        except (KeyError, TypeError, ValueError):
            return False
        return expected is None or expected == Checkpoint(len(records), previous)

