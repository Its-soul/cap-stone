#!/usr/bin/env python3
"""Phase 6 frozen protocol evaluation runner & table rebuilder.

Supports:
  --run-offline          Run 104-claim x 10-seed offline track, 5 ablations, and 32 update paths
  --run-live             Run stratified 36-claim Option-B live Groq track (resumable via --resume)
  --rebuild-tables-only  Regenerate JSON & Markdown tables strictly from saved raw JSONL logs
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cert_recovery.oracle.benchmark_v1 import load_blind_external_cases  # noqa: E402
from cert_recovery.oracle.benchmark_eval import (  # noqa: E402
    _load_jsonl,
    _resolve_repo_path,
    rebuild_phase6_tables,
    run_live_phase6,
    run_offline_phase6,
    select_live_subset_36,
    verify_frozen_manifest,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute Phase 6 frozen evaluation and rebuild tables.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repository root for resolving relative manifest paths")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "results" / "phase5" / "frozen_evaluation_manifest.json")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results" / "phase6")
    parser.add_argument("--blind-path", type=Path, default=REPO_ROOT / "datasets" / "blind_external_template.json")
    parser.add_argument("--run-offline", action="store_true", help="Run 104x10 offline evaluation, ablations, and recovery")
    parser.add_argument("--run-live", action="store_true", help="Run Option-B stratified 36-claim live evaluation")
    parser.add_argument("--rebuild-tables-only", action="store_true", help="Rebuild tables only from saved raw JSONL logs")
    parser.add_argument("--provider", type=str, default="groq")
    parser.add_argument("--model-id", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--api-key-env", type=str, default="GROQ_API_KEY")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--min-call-interval-sec", type=float, default=5.0)
    parser.add_argument("--max-calls", type=int, default=320)
    parser.add_argument("--max-cost-usd", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true", help="Resume live runs skipping completed records in raw JSONL")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    results_dir = args.results_dir if args.results_dir.is_absolute() else (repo_root / args.results_dir)
    raw_dir = results_dir / "raw"
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Mandatory SHA-256 pre-flight before ANY Phase 6 evaluation or rebuild
    try:
        manifest_info = verify_frozen_manifest(args.manifest, repo_root)
    except Exception as exc:
        print(f"FROZEN MANIFEST PRE-FLIGHT FAILED: {exc}", file=sys.stderr)
        return 2

    blind_info = load_blind_external_cases(args.blind_path)
    blind_arm_status = (
        "NOT RUN (0 submitted)"
        if blind_info["status"] == "PENDING_TEAMMATE_SUBMISSION"
        else f"READY ({blind_info['count']} submitted)"
    )

    hashes = manifest_info["manifest"]["hashes"]
    test_inputs = _load_jsonl(_resolve_repo_path(hashes["test_inputs"]["path"], repo_root))
    test_labels = _load_jsonl(_resolve_repo_path(hashes["test_oracle_labels"]["path"], repo_root))
    upd_inputs = _load_jsonl(_resolve_repo_path(hashes["test_update_inputs"]["path"], repo_root))
    upd_labels = _load_jsonl(_resolve_repo_path(hashes["test_update_oracle_labels"]["path"], repo_root))

    subset_36 = select_live_subset_36(test_inputs, test_labels, upd_inputs)
    (results_dir / "live_subset_36_manifest.json").write_text(
        json.dumps({k: v for k, v in subset_36.items() if k not in ("claims", "updates")}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    attempt_manifest_path = results_dir / "attempt_manifest.json"
    prev_attempt = json.loads(attempt_manifest_path.read_text(encoding="utf-8")) if attempt_manifest_path.exists() else {}

    offline_summary = prev_attempt.get("offline_track", {"status": "NOT RUN"})
    live_summary = prev_attempt.get("live_track", {"status": "NOT RUN"})

    if not args.rebuild_tables_only:
        if args.run_offline:
            offline_summary = run_offline_phase6(
                manifest_info=manifest_info,
                repo_root=repo_root,
                raw_dir=raw_dir,
            )
            offline_summary = {"status": "PASS", **offline_summary}

        if args.run_live:
            live_summary = run_live_phase6(
                subset_36=subset_36,
                test_labels_by_id={r["claim_id"]: r["label"] for r in test_labels},
                upd_labels_by_id={r["transition_id"]: r for r in upd_labels},
                raw_dir=raw_dir,
                provider=args.provider,
                model_id=args.model_id,
                api_key_env_var=args.api_key_env,
                base_url=args.base_url,
                min_call_interval_sec=args.min_call_interval_sec,
                max_calls=args.max_calls,
                max_cost_usd=args.max_cost_usd,
                resume=args.resume,
            )
        elif live_summary.get("status") == "NOT RUN":
            # Explicitly check whether live key exists right now so attempt manifest records BLOCKED if missing
            from cert_recovery.adjudication.model_agent import HostedChatModelClient
            probe = HostedChatModelClient(provider=args.provider, api_key_env_var=args.api_key_env)
            if not probe.is_configured():
                live_summary = {
                    "status": "BLOCKED",
                    "reason": f"Live inference not run: '{args.api_key_env}' environment variable is unset.",
                    "calls_used": 0,
                    "tokens_used": 0,
                    "completed_claims": 0,
                }

    attempt_manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_manifest_path": str(args.manifest),
        "hash_preflight": {
            "status": "PASS",
            "verified_files_count": len(manifest_info["verified_hashes"]),
            "verified_hashes": manifest_info["verified_hashes"],
        },
        "blind_external_arm": blind_arm_status,
        "offline_track": offline_summary,
        "live_track": live_summary,
        "mode": "rebuild_tables_only" if args.rebuild_tables_only else "evaluate",
    }
    attempt_manifest_path.write_text(json.dumps(attempt_manifest, indent=2, sort_keys=True), encoding="utf-8")

    tables = rebuild_phase6_tables(results_dir=results_dir, manifest_info=manifest_info, repo_root=repo_root)
    print(json.dumps({
        "hash_preflight": "PASS",
        "blind_external_arm": blind_arm_status,
        "offline_status": offline_summary.get("status"),
        "live_status": live_summary.get("status"),
        "live_reason": live_summary.get("reason"),
        "recovery_overall_equivalence": tables.get("recovery_overall_equivalence"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
