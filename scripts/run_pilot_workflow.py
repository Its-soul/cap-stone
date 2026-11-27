#!/usr/bin/env python3
"""Run Phase 4 role-isolated model specialist pilot and 5-class fault matrix.

By default runs in strict gate mode: if provider, model ID (>=7B / hosted API),
or API key environment variable is unset/unavailable, reports live execution as
`BLOCKED` (never fakes live inference) while verifying the full adapter contract,
Phase 2 independent verifier, Phase 3 signed provenance, and 5-class fault
matrix offline with explicitly labeled fakes.
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

from cert_recovery.adjudication.pilot_runner import execute_phase4_suite  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 4 pilot and bounded 5-class fault matrix.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase4"))
    parser.add_argument("--provider", type=str, default=None, help="openai_compatible, groq, or gemini")
    parser.add_argument("--model-id", type=str, default=None, help="Hosted API or >=7B model ID")
    parser.add_argument(
        "--api-key-env",
        type=str,
        default=None,
        help="Env var holding API key (defaults to GROQ_API_KEY when --provider groq; never logged)",
    )
    parser.add_argument("--base-url", type=str, default=None, help="Optional custom OpenAI-compatible/Groq/Gemini URL")
    parser.add_argument("--max-calls", type=int, default=60, help="Hard call budget cap")
    parser.add_argument("--max-cost-usd", type=float, default=1.0, help="Hard USD budget cap")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--min-call-interval-sec",
        type=float,
        default=0.0,
        help="Optional minimum interval in seconds between outbound hosted API calls for TPM/RPM pacing",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    api_key_env = args.api_key_env or ("GROQ_API_KEY" if args.provider == "groq" else None)
    summary = execute_phase4_suite(
        args.output_dir,
        provider=args.provider,
        model_id=args.model_id,
        api_key_env_var=api_key_env,
        base_url=args.base_url,
        max_calls=args.max_calls,
        max_cost_usd=args.max_cost_usd,
        temperature=args.temperature,
        min_call_interval_sec=args.min_call_interval_sec,
    )
    print(json.dumps({
        "live_execution_status": summary["live_execution"]["status"],
        "live_blocker_reason": summary["live_execution"]["blocker_reason"],
        "offline_contract_status": summary["offline_contract_verification"]["status"],
        "fault_scenarios_passed": summary["offline_contract_verification"]["fault_scenarios_passed"],
        "fault_scenarios_total": summary["offline_contract_verification"]["fault_scenarios_total"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
