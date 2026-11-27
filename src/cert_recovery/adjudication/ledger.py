"""Content-addressed event DAG with optional Ed25519 signatures and JSONL persistence."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any
import uuid

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from ..crypto import canonical_json, sha256
from ..models import utc_now
from .provenance_verifier import (
    GENESIS_HASH,
    SIGNED_CHECKPOINT_SCHEMA,
    SIGNED_EVENT_SCHEMA,
    canonical_checkpoint_signing_bytes,
    canonical_event_signing_bytes,
    compute_artifact_version_map,
    compute_key_id,
    load_events_jsonl,
    verify_signed_records,
)


class Ed25519EventSigner:
    """In-memory Ed25519 signer (private key kept strictly out of repo/logs)."""

    def __init__(self, private_key: Ed25519PrivateKey | None = None) -> None:
        self._private_key = private_key or Ed25519PrivateKey.generate()
        self.public_key: Ed25519PublicKey = self._private_key.public_key()
        self.key_id: str = compute_key_id(self.public_key)

    @classmethod
    def generate(cls) -> Ed25519EventSigner:
        return cls(Ed25519PrivateKey.generate())

    def public_key_bytes(self) -> bytes:
        return self.public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)

    def public_key_hex(self) -> str:
        return self.public_key_bytes().hex()

    def sign_bytes(self, data: bytes) -> str:
        return self._private_key.sign(data).hex()

    def write_public_key_hex(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.public_key_hex() + "\n", encoding="utf-8")
        return target


def _fsync_write_atomic(path: Path, content: str) -> None:
    """Write checkpoint atomically via temp file + flush/fsync + close + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        for attempt in range(8):
            try:
                os.replace(tmp_path, path)
                return
            except PermissionError:
                # Transient Windows file-indexer/AV lock during rapid sequential checkpoint updates
                if attempt == 7:
                    raise
                time.sleep(0.005 * (attempt + 1))
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


class EventDAG:
    """Content-addressed event DAG, supporting both legacy unsigned and Ed25519 persisted modes."""

    def __init__(
        self,
        clock=utc_now,
        *,
        run_id: str | None = None,
        signer: Ed25519EventSigner | None = None,
        log_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        auto_flush: bool = True,
    ) -> None:
        self._records: list[dict[str, Any]] = []
        self._clock = clock
        self.signer = signer
        self.log_path = Path(log_path) if log_path is not None else None
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.auto_flush = auto_flush
        self.run_id = run_id or (f"run-{uuid.uuid4().hex[:12]}" if (signer or log_path or checkpoint_path) else None)

        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if self.log_path.exists() and self.log_path.stat().st_size > 0:
                if self.signer is None:
                    raise ValueError("Reloading an existing persisted log requires a signer or EventDAG.open")
                existing = load_events_jsonl(self.log_path)
                expected_ckpt = (
                    json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
                    if (self.checkpoint_path is not None and self.checkpoint_path.exists())
                    else None
                )
                vr = verify_signed_records(existing, self.signer.public_key, expected_ckpt)
                if not vr.valid:
                    raise ValueError(f"Persisted provenance log failed verification on reload: {vr.reason}")
                if existing and existing[0]["run_id"] != self.run_id:
                    raise ValueError("Reloaded log run_id does not match configured run_id")
                self._records = existing

    @classmethod
    def open_persisted(
        cls,
        provenance_dir: str | Path,
        *,
        run_id: str | None = None,
        signer: Ed25519EventSigner | None = None,
        clock=utc_now,
    ) -> EventDAG:
        base = Path(provenance_dir)
        base.mkdir(parents=True, exist_ok=True)
        events_path = base / "events.jsonl"
        ckpt_path = base / "checkpoint.json"
        if events_path.exists() and run_id is None:
            existing = load_events_jsonl(events_path)
            if existing:
                run_id = existing[0]["run_id"]
        if signer is None:
            signer = Ed25519EventSigner.generate()
        return cls(
            clock=clock,
            run_id=run_id,
            signer=signer,
            log_path=events_path,
            checkpoint_path=ckpt_path,
        )

    @property
    def is_signed(self) -> bool:
        return self.signer is not None

    def append(
        self,
        event_type: str,
        payload: Any,
        parents: tuple[str, ...] | list[str] = (),
        agent_id: str | None = None,
        *,
        artifact_ref: str | None = None,
        artifact_version: dict[str, Any] | None = None,
    ) -> str:
        known = {r["event_hash"] for r in self._records}
        parent_list = sorted(set(parents))
        if not set(parent_list) <= known:
            raise ValueError("Parents must precede event")

        normalized_payload = json.loads(canonical_json(payload))
        payload_digest = sha256(payload)
        seq = len(self._records)

        if self.signer is None:
            body = {
                "event_id": seq,
                "event_type": event_type,
                "agent_id": agent_id,
                "timestamp": self._clock(),
                "payload": normalized_payload,
                "payload_hash": payload_digest,
                "parents": parent_list,
            }
            record = {**body, "event_hash": sha256(body)}
            self._records.append(record)
            return record["event_hash"]

        body = {
            "schema_version": SIGNED_EVENT_SCHEMA,
            "run_id": self.run_id,
            "sequence": seq,
            "event_id": seq,
            "event_type": event_type,
            "agent_id": agent_id,
            "timestamp": self._clock(),
            "artifact_ref": artifact_ref,
            "artifact_version": artifact_version,
            "payload": normalized_payload,
            "payload_digest": payload_digest,
            "parent_hashes": parent_list,
            "signer_key_id": self.signer.key_id,
        }
        event_hash = sha256(body)
        record = {**body, "event_hash": event_hash}
        record["signature"] = self.signer.sign_bytes(canonical_event_signing_bytes(record))

        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(canonical_json(record) + "\n")
                if self.auto_flush:
                    fh.flush()
                    os.fsync(fh.fileno())

        self._records.append(record)
        if self.checkpoint_path is not None and self.auto_flush:
            self.write_checkpoint(self.checkpoint_path)
        return event_hash

    def write_checkpoint(self, path: str | Path | None = None) -> dict[str, Any]:
        target = Path(path) if path is not None else self.checkpoint_path
        ckpt = self.checkpoint()
        if target is not None:
            _fsync_write_atomic(target, json.dumps(ckpt, indent=2, sort_keys=True) + "\n")
        return ckpt

    def export(self) -> list[dict[str, Any]]:
        return json.loads(canonical_json(self._records))

    def checkpoint(self) -> dict[str, Any]:
        if self.signer is None:
            return self.aggregate(self._records)
        return self.build_signed_checkpoint(self._records, self.run_id, self.signer)

    @staticmethod
    def build_signed_checkpoint(
        records: list[dict[str, Any]],
        run_id: str | None,
        signer: Ed25519EventSigner,
    ) -> dict[str, Any]:
        parents = {p for r in records for p in r["parent_hashes"]}
        tips = sorted(r["event_hash"] for r in records if r["event_hash"] not in parents)
        ckpt = {
            "schema_version": SIGNED_CHECKPOINT_SCHEMA,
            "run_id": run_id or (records[0]["run_id"] if records else ""),
            "head_sequence": records[-1]["sequence"] if records else -1,
            "head_event_hash": records[-1]["event_hash"] if records else GENESIS_HASH,
            "event_count": len(records),
            "tips": tips,
            "root_hash": sha256({"tips": tips}),
            "artifact_version_to_event_id": compute_artifact_version_map(records),
            "signer_key_id": signer.key_id,
        }
        ckpt["checkpoint_signature"] = signer.sign_bytes(canonical_checkpoint_signing_bytes(ckpt))
        return ckpt

    @staticmethod
    def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
        if records and "parent_hashes" in records[0]:
            parents = {p for r in records for p in r["parent_hashes"]}
            tips = sorted(r["event_hash"] for r in records if r["event_hash"] not in parents)
            return {"length": len(records), "root_hash": sha256({"tips": tips}), "tips": tips}
        parents = {p for r in records for p in r["parents"]}
        tips = sorted(r["event_hash"] for r in records if r["event_hash"] not in parents)
        return {"length": len(records), "root_hash": sha256({"tips": tips}), "tips": tips}

    @staticmethod
    def verify(records: list[dict[str, Any]], expected: dict[str, Any] | None = None, public_key=None) -> bool:
        if records and "signature" in records[0]:
            if public_key is None:
                return False
            return verify_signed_records(records, public_key, expected).valid
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

    def trace(self, event_hash: str) -> list[dict[str, Any]]:
        by_hash = {r["event_hash"]: r for r in self.export()}
        pending, seen = [event_hash], set()
        while pending:
            current = pending.pop()
            if current not in seen:
                seen.add(current)
                rec = by_hash[current]
                parents = rec["parent_hashes"] if "parent_hashes" in rec else rec["parents"]
                pending.extend(parents)
        return [r for r in by_hash.values() if r["event_hash"] in seen]
