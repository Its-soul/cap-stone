"""Content-addressed event DAG alongside (not replacing) the engine audit chain."""
import json

from ..crypto import canonical_json, sha256
from ..models import utc_now


class EventDAG:
    def __init__(self, clock=utc_now):
        self._records = []
        self._clock = clock

    def append(self, event_type, payload, parents=(), agent_id=None):
        known = {r["event_hash"] for r in self._records}
        parents = sorted(set(parents))
        if not set(parents) <= known:
            raise ValueError("Parents must precede event")
        body = {"event_id": len(self._records), "event_type": event_type,
                "agent_id": agent_id, "timestamp": self._clock(),
                "payload": json.loads(canonical_json(payload)), "payload_hash": sha256(payload),
                "parents": parents}
        record = {**body, "event_hash": sha256(body)}
        self._records.append(record)
        return record["event_hash"]

    def export(self):
        return json.loads(canonical_json(self._records))

    def checkpoint(self):
        return self.aggregate(self._records)

    @staticmethod
    def aggregate(records):
        parents = {p for r in records for p in r["parents"]}
        tips = sorted(r["event_hash"] for r in records if r["event_hash"] not in parents)
        return {"length": len(records), "root_hash": sha256({"tips": tips}), "tips": tips}

    @staticmethod
    def verify(records, expected=None):
        seen = set()
        try:
            for index, record in enumerate(records):
                if set(record) != {"event_id", "event_type", "agent_id", "timestamp", "payload", "payload_hash", "parents", "event_hash"}:
                    return False
                if (type(record["event_id"]) is not int or record["event_id"] != index
                        or not isinstance(record["parents"], list)
                        or record["parents"] != sorted(set(record["parents"]))
                        or not set(record["parents"]) <= seen
                        or record["payload_hash"] != sha256(record["payload"])
                        or record["event_hash"] != sha256({k: v for k, v in record.items() if k != "event_hash"})):
                    return False
                seen.add(record["event_hash"])
            return expected is None or expected == EventDAG.aggregate(records)
        except (TypeError, ValueError, KeyError):
            return False

    def trace(self, event_hash):
        by_hash = {r["event_hash"]: r for r in self.export()}
        pending, seen = [event_hash], set()
        while pending:
            current = pending.pop()
            if current not in seen:
                seen.add(current)
                pending.extend(by_hash[current]["parents"])
        return [r for r in by_hash.values() if r["event_hash"] in seen]
