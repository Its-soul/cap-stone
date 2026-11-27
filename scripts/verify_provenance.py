"""Standalone CLI to verify persisted Ed25519-signed EventDAG provenance logs.

Usage:
    python scripts/verify_provenance.py --events path/to/events.jsonl --public-key path/to/public_key.hex [--checkpoint path/to/checkpoint.json] [--query-artifact facts:evidence@1] [--json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cert_recovery.adjudication.provenance_verifier import (  # noqa: E402
    load_events_jsonl,
    query_decisions_for_artifact,
    verify_persisted_run,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify persisted Ed25519-signed EventDAG logs and optional checkpoints."
    )
    parser.add_argument("--events", required=True, type=Path, help="Path to events.jsonl")
    parser.add_argument(
        "--public-key",
        required=True,
        type=str,
        help="Path to Ed25519 public_key.hex / .json or 64-char hex string",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional independently retained checkpoint.json to verify against the log",
    )
    parser.add_argument(
        "--query-artifact",
        type=str,
        default=None,
        help="Optional causal lineage query, e.g. facts:evidence@1 (run only when valid)",
    )
    parser.add_argument(
        "--include-operational-edges",
        action="store_true",
        help="Include operational recovery_started/recovery edges in --query-artifact lineage query",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = verify_persisted_run(args.events, args.public_key, args.checkpoint)
    payload = result.to_dict()

    if args.query_artifact is not None:
        if result.valid:
            records = load_events_jsonl(args.events)
            deps = sorted(
                query_decisions_for_artifact(
                    records,
                    args.query_artifact,
                    include_operational_edges=args.include_operational_edges,
                )
            )
            payload["lineage_query"] = {
                "target_artifact_ref": args.query_artifact,
                "include_operational_edges": args.include_operational_edges,
                "dependent_decisions": deps,
            }
        else:
            payload["lineage_query"] = {
                "target_artifact_ref": args.query_artifact,
                "error": "Skipped lineage query because provenance verification failed",
            }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        status = "PASS" if result.valid else "FAIL"
        print(f"PROVENANCE VERIFICATION: {status}")
        print(f"  reason:              {result.reason}")
        print(f"  error_code:          {result.error_code}")
        print(f"  run_id:              {result.run_id}")
        print(f"  event_count:         {result.event_count}")
        print(f"  head_sequence:       {result.head_sequence}")
        print(f"  head_event_hash:     {result.head_event_hash}")
        print(f"  signer_key_id:       {result.signer_key_id}")
        print(f"  checkpoint_verified: {result.checkpoint_verified}")
        if "lineage_query" in payload:
            print(f"  lineage_query:       {payload['lineage_query']}")
        print("  established_by_signatures_and_retained_checkpoint:")
        for item in payload["trust_model"]["established_by_signatures_and_retained_checkpoint"]:
            print(f"    - {item}")
        print("  not_established:")
        for item in payload["trust_model"]["not_established"]:
            print(f"    - {item}")

    return 0 if result.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
