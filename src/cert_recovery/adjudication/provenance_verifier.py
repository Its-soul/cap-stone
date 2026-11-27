"""Standalone Ed25519 verification, trust-model spec, and causal lineage queries.

Clean runtime module: imports ZERO modules from `cert_recovery.oracle` or
evaluation harnesses. Verifies persisted `events.jsonl` logs and optional
independently retained `checkpoint.json` anchors.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from ..crypto import canonical_json, sha256

SIGNED_EVENT_SCHEMA = "signed-event-dag-v1"
SIGNED_CHECKPOINT_SCHEMA = "signed-checkpoint-v1"
EVENT_SIGNATURE_DOMAIN = "cert-recovery.event.v1"
CHECKPOINT_SIGNATURE_DOMAIN = "cert-recovery.checkpoint.v1"
GENESIS_HASH = "0" * 64

SIGNED_EVENT_KEYS = frozenset({
    "schema_version",
    "run_id",
    "sequence",
    "event_id",
    "event_type",
    "agent_id",
    "timestamp",
    "artifact_ref",
    "artifact_version",
    "payload",
    "payload_digest",
    "parent_hashes",
    "signer_key_id",
    "event_hash",
    "signature",
})

SIGNED_CHECKPOINT_KEYS = frozenset({
    "schema_version",
    "run_id",
    "head_sequence",
    "head_event_hash",
    "event_count",
    "tips",
    "root_hash",
    "artifact_version_to_event_id",
    "signer_key_id",
    "checkpoint_signature",
})

TRUST_MODEL_SUMMARY: dict[str, Any] = {
    "established_by_signatures_and_retained_checkpoint": [
        "Tamper evidence: any byte change to payload, parents, sequence, run_id, or artifact_ref invalidates SHA-256 digests and Ed25519 signatures.",
        "Signing-key possession: every accepted event and checkpoint was signed by the holder of the Ed25519 private key matching the trusted public key.",
        "Causal ordering & interior completeness: parent_hashes must precede children in monotonic sequence with zero gaps or dangling links.",
        "Cross-run isolation: run_id binding prevents injecting validly signed events from another run.",
        "Truncation & rollback detection (WITH retained checkpoint): comparing against an independently retained checkpoint.json detects tail truncation and rollback to an older checkpoint.",
    ],
    "not_established": [
        "Real-world identity beyond possession of the trusted Ed25519 private key (no external PKI/CA binding).",
        "Ground-truth or factual accuracy of source evidence payloads (garbage-in still signs cleanly).",
        "Multi-node or Byzantine quorum consensus (single-writer local log).",
        "Tail truncation or rollback detection WITHOUT an independently retained checkpoint (a truncated prefix remains internally valid).",
        "Crash-atomic single transaction across two files: a crash between events.jsonl fsync and checkpoint.json os.replace leaves a recoverable lagging-checkpoint gap.",
    ],
}


def compute_key_id(public_key: Ed25519PublicKey) -> str:
    """Return stable 16-hex-char key fingerprint of raw 32-byte Ed25519 public key."""
    raw = public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
    return sha256({"ed25519_raw_hex": raw.hex()})[:16]


def load_public_key(source: str | Path | bytes | Ed25519PublicKey) -> Ed25519PublicKey:
    """Load an Ed25519 public key from a key object, raw bytes, hex string, or file path."""
    if isinstance(source, Ed25519PublicKey):
        return source
    if isinstance(source, bytes):
        return Ed25519PublicKey.from_public_bytes(source)
    if isinstance(source, Path) or (isinstance(source, str) and Path(source).exists()):
        path = Path(source)
        text = path.read_text(encoding="utf-8").strip()
        if path.suffix == ".json":
            data = json.loads(text)
            text = data["public_key_hex"]
        return Ed25519PublicKey.from_public_bytes(bytes.fromhex(text))
    if isinstance(source, str):
        cleaned = source.strip()
        if len(cleaned) == 64:
            return Ed25519PublicKey.from_public_bytes(bytes.fromhex(cleaned))
    raise ValueError("Unsupported public key format; supply Ed25519PublicKey, 64-char hex, or existing file path")


def canonical_event_body(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": record["schema_version"],
        "run_id": record["run_id"],
        "sequence": record["sequence"],
        "event_id": record["event_id"],
        "event_type": record["event_type"],
        "agent_id": record["agent_id"],
        "timestamp": record["timestamp"],
        "artifact_ref": record["artifact_ref"],
        "artifact_version": record["artifact_version"],
        "payload": record["payload"],
        "payload_digest": record["payload_digest"],
        "parent_hashes": record["parent_hashes"],
        "signer_key_id": record["signer_key_id"],
    }


def canonical_event_signing_bytes(record: dict[str, Any]) -> bytes:
    body = canonical_event_body(record)
    return canonical_json({"domain": EVENT_SIGNATURE_DOMAIN, **body, "event_hash": record["event_hash"]}).encode("utf-8")


def canonical_checkpoint_signing_bytes(checkpoint: dict[str, Any]) -> bytes:
    body = {k: checkpoint[k] for k in sorted(SIGNED_CHECKPOINT_KEYS - {"checkpoint_signature"})}
    return canonical_json({"domain": CHECKPOINT_SIGNATURE_DOMAIN, **body}).encode("utf-8")


def compute_artifact_version_map(records: list[dict[str, Any]]) -> dict[str, int]:
    """Build deterministic artifact_ref -> canonical producing event_id mapping."""
    preferred_types = {
        "input_received",
        "specialist_assessment",
        "fault_assessment",
        "specialist_composition",
        "consensus",
        "final_decision",
    }
    mapping: dict[str, int] = {}
    for rec in records:
        art_ref = rec.get("artifact_ref")
        if not art_ref:
            continue
        if art_ref not in mapping or rec.get("event_type") in preferred_types:
            mapping[art_ref] = rec["event_id"]
    return dict(sorted(mapping.items()))


def load_events_jsonl(events_path: str | Path) -> list[dict[str, Any]]:
    path = Path(events_path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, 1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Corrupt JSON on line {line_no} of {path}: {exc}") from exc
    return records


@dataclass(frozen=True)
class ProvenanceVerificationResult:
    valid: bool
    reason: str
    error_code: str | None
    run_id: str | None
    event_count: int
    head_sequence: int
    head_event_hash: str
    signer_key_id: str | None
    checkpoint_verified: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "error_code": self.error_code,
            "run_id": self.run_id,
            "event_count": self.event_count,
            "head_sequence": self.head_sequence,
            "head_event_hash": self.head_event_hash,
            "signer_key_id": self.signer_key_id,
            "checkpoint_verified": self.checkpoint_verified,
            "trust_model": TRUST_MODEL_SUMMARY,
        }


def verify_signed_records(
    records: list[dict[str, Any]],
    public_key: Ed25519PublicKey | str | Path | bytes,
    expected_checkpoint: dict[str, Any] | None = None,
) -> ProvenanceVerificationResult:
    """Verify signed EventDAG records and optional independently retained checkpoint."""
    try:
        pub = load_public_key(public_key)
    except Exception as exc:
        return ProvenanceVerificationResult(
            False, f"Invalid trusted public key: {exc}", "invalid_public_key", None, len(records), -1, GENESIS_HASH, None, False
        )
    trusted_key_id = compute_key_id(pub)

    if expected_checkpoint is not None:
        if not isinstance(expected_checkpoint, dict) or set(expected_checkpoint) != SIGNED_CHECKPOINT_KEYS:
            return ProvenanceVerificationResult(
                False, "Malformed expected checkpoint schema", "malformed_checkpoint", None, len(records), -1, GENESIS_HASH, trusted_key_id, False
            )
        if expected_checkpoint.get("schema_version") != SIGNED_CHECKPOINT_SCHEMA:
            return ProvenanceVerificationResult(
                False, "Unsupported checkpoint schema version", "malformed_checkpoint", None, len(records), -1, GENESIS_HASH, trusted_key_id, False
            )
        if expected_checkpoint.get("signer_key_id") != trusted_key_id:
            return ProvenanceVerificationResult(
                False, "Checkpoint signer_key_id does not match trusted public key", "untrusted_checkpoint_signer", expected_checkpoint.get("run_id"), len(records), -1, GENESIS_HASH, trusted_key_id, False
            )
        try:
            sig_bytes = bytes.fromhex(expected_checkpoint["checkpoint_signature"])
            pub.verify(sig_bytes, canonical_checkpoint_signing_bytes(expected_checkpoint))
        except (ValueError, InvalidSignature):
            return ProvenanceVerificationResult(
                False, "Checkpoint Ed25519 signature verification failed", "invalid_checkpoint_signature", expected_checkpoint.get("run_id"), len(records), -1, GENESIS_HASH, trusted_key_id, False
            )

    seen_hashes: set[str] = set()
    run_id: str | None = expected_checkpoint.get("run_id") if expected_checkpoint else None
    head_hash = GENESIS_HASH
    head_seq = -1

    for idx, rec in enumerate(records):
        if not isinstance(rec, dict) or set(rec) != SIGNED_EVENT_KEYS:
            return ProvenanceVerificationResult(
                False, f"Malformed event schema at index {idx}", "malformed_event_schema", run_id, len(records), head_seq, head_hash, trusted_key_id, False
            )
        if rec["schema_version"] != SIGNED_EVENT_SCHEMA:
            return ProvenanceVerificationResult(
                False, f"Unsupported event schema at index {idx}", "malformed_event_schema", run_id, len(records), head_seq, head_hash, trusted_key_id, False
            )
        if type(rec["sequence"]) is not int or rec["sequence"] != idx or rec["event_id"] != idx:
            return ProvenanceVerificationResult(
                False, f"Non-monotonic sequence/event_id at index {idx}", "sequence_mismatch", run_id, len(records), head_seq, head_hash, trusted_key_id, False
            )
        if run_id is None:
            run_id = rec["run_id"]
        elif rec["run_id"] != run_id:
            return ProvenanceVerificationResult(
                False, f"Cross-run event detected at sequence {idx}: {rec['run_id']} != {run_id}", "run_id_mismatch", run_id, len(records), head_seq, head_hash, trusted_key_id, False
            )
        if rec["signer_key_id"] != trusted_key_id:
            return ProvenanceVerificationResult(
                False, f"Untrusted signer_key_id at sequence {idx}: {rec['signer_key_id']}", "untrusted_signer_key", run_id, len(records), head_seq, head_hash, trusted_key_id, False
            )

        parents = rec["parent_hashes"]
        if not isinstance(parents, list) or parents != sorted(set(parents)) or not set(parents) <= seen_hashes:
            return ProvenanceVerificationResult(
                False, f"Invalid or missing parent_hashes at sequence {idx}", "parent_link_error", run_id, len(records), head_seq, head_hash, trusted_key_id, False
            )

        try:
            computed_payload_digest = sha256(rec["payload"])
        except ValueError as exc:
            return ProvenanceVerificationResult(
                False, f"Non-canonical payload at sequence {idx}: {exc}", "payload_digest_error", run_id, len(records), head_seq, head_hash, trusted_key_id, False
            )
        if rec["payload_digest"] != computed_payload_digest:
            return ProvenanceVerificationResult(
                False, f"Payload digest mismatch at sequence {idx}", "payload_digest_mismatch", run_id, len(records), head_seq, head_hash, trusted_key_id, False
            )

        computed_event_hash = sha256(canonical_event_body(rec))
        if rec["event_hash"] != computed_event_hash:
            return ProvenanceVerificationResult(
                False, f"Event hash mismatch at sequence {idx}", "event_hash_mismatch", run_id, len(records), head_seq, head_hash, trusted_key_id, False
            )

        try:
            sig_bytes = bytes.fromhex(rec["signature"])
            pub.verify(sig_bytes, canonical_event_signing_bytes(rec))
        except (ValueError, InvalidSignature):
            return ProvenanceVerificationResult(
                False, f"Ed25519 signature verification failed at sequence {idx}", "invalid_ed25519_signature", run_id, len(records), head_seq, head_hash, trusted_key_id, False
            )

        seen_hashes.add(rec["event_hash"])
        head_hash = rec["event_hash"]
        head_seq = idx

    parent_set = {p for r in records for p in r["parent_hashes"]}
    tips = sorted(r["event_hash"] for r in records if r["event_hash"] not in parent_set)
    root_hash = sha256({"tips": tips})
    art_map = compute_artifact_version_map(records)

    checkpoint_verified = False
    if expected_checkpoint is not None:
        if expected_checkpoint["event_count"] != len(records):
            return ProvenanceVerificationResult(
                False,
                f"Checkpoint event_count mismatch: log has {len(records)}, expected {expected_checkpoint['event_count']}",
                "checkpoint_event_count_mismatch",
                run_id,
                len(records),
                head_seq,
                head_hash,
                trusted_key_id,
                False,
            )
        if (
            expected_checkpoint["head_sequence"] != head_seq
            or expected_checkpoint["head_event_hash"] != head_hash
            or expected_checkpoint["tips"] != tips
            or expected_checkpoint["root_hash"] != root_hash
            or expected_checkpoint["artifact_version_to_event_id"] != art_map
        ):
            return ProvenanceVerificationResult(
                False,
                "Checkpoint head/tips/artifact-map mismatch against persisted log",
                "checkpoint_head_mismatch",
                run_id,
                len(records),
                head_seq,
                head_hash,
                trusted_key_id,
                False,
            )
        checkpoint_verified = True

    return ProvenanceVerificationResult(
        True,
        "All event digests, causal parent links, and Ed25519 signatures verified"
        + (" (checkpoint matched)" if checkpoint_verified else " (no retained checkpoint provided)"),
        None,
        run_id,
        len(records),
        head_seq,
        head_hash,
        trusted_key_id,
        checkpoint_verified,
    )


def verify_persisted_run(
    events_path: str | Path,
    public_key: Ed25519PublicKey | str | Path | bytes,
    checkpoint_path: str | Path | None = None,
) -> ProvenanceVerificationResult:
    """Verify a persisted `events.jsonl` file and optional `checkpoint.json` from disk."""
    try:
        records = load_events_jsonl(events_path)
    except Exception as exc:
        return ProvenanceVerificationResult(
            False, f"Failed to read events log: {exc}", "unreadable_events_log", None, 0, -1, GENESIS_HASH, None, False
        )
    expected_checkpoint = None
    if checkpoint_path is not None:
        try:
            expected_checkpoint = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
        except Exception as exc:
            return ProvenanceVerificationResult(
                False, f"Failed to read checkpoint: {exc}", "unreadable_checkpoint", None, len(records), -1, GENESIS_HASH, None, False
            )
    return verify_signed_records(records, public_key, expected_checkpoint)


def query_decisions_for_artifact(
    records: list[dict[str, Any]],
    target_artifact_ref: str,
    *,
    include_operational_edges: bool = False,
) -> set[str]:
    """Return decision artifact_refs causally reachable from `target_artifact_ref`.

    By default, traverses data-dependency lineage only (stopping at `recovery_started`
    and `recovery` operational lifecycle nodes so rebuilt decisions grounded on `v2`
    inputs are not misattributed to `v1` inputs). Set `include_operational_edges=True`
    to inspect raw EventDAG reachability across operational recovery links.
    """
    children_by_hash: dict[str, list[dict[str, Any]]] = {}
    root_hashes: list[str] = []
    for rec in records:
        parents = rec.get("parent_hashes") if "parent_hashes" in rec else rec.get("parents", [])
        for p in parents:
            children_by_hash.setdefault(p, []).append(rec)
        if rec.get("artifact_ref") == target_artifact_ref and rec.get("event_type") == "input_received":
            root_hashes.append(rec["event_hash"])
        elif not root_hashes and rec.get("event_type") == "input_received":
            binding = rec.get("payload", {}).get("binding", {})
            if f"{binding.get('artifact_id')}@{binding.get('version')}" == target_artifact_ref:
                root_hashes.append(rec["event_hash"])

    decisions: set[str] = set()
    seen: set[str] = set()
    pending = list(root_hashes)
    while pending:
        cur = pending.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for child in children_by_hash.get(cur, []):
            etype = child.get("event_type")
            if not include_operational_edges and etype in ("recovery_started", "recovery"):
                continue
            if etype == "final_decision":
                dec_ref = child.get("artifact_ref")
                if not dec_ref:
                    cert = child.get("payload", {}).get("certificate", {})
                    if "version" in cert:
                        dec_ref = f"final@{cert['version']}"
                if dec_ref:
                    decisions.add(dec_ref)
            pending.append(child["event_hash"])
    return decisions
