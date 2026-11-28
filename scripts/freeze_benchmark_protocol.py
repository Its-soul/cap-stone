#!/usr/bin/env python3
"""Freeze Phase 5 specification, 128-claim isolated splits, oracle labels, and manifest.

STRICT RULE: This script ONLY builds, audits, and hashes frozen splits, oracle labels,
and baseline/statistical specifications. It does NOT execute ANY model or adjudication
system on the frozen test set (that belongs strictly to Phase 6).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cert_recovery.oracle.benchmark_v1 import (  # noqa: E402
    GENERATOR_SEED,
    GENERATOR_VERSION,
    SPEC_VERSION,
    SWEEP_SEEDS,
    audit_frozen_dataset,
    generate_frozen_splits,
    load_blind_external_cases,
)
from cert_recovery.oracle.protocol_v1 import (  # noqa: E402
    DEPRECATED_HISTORICAL_ARM_NOTE,
    PRIMARY_BASELINES_V1,
    TARGETED_ABLATIONS_V1,
    build_groq_8k_tpm_budget_plan,
    frozen_statistical_protocol,
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _git_head() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
        return out
    except Exception:
        return "UNKNOWN"


def freeze_phase5(
    *,
    dataset_dir: Path,
    results_dir: Path,
    spec_path: Path = REPO_ROOT / "docs" / "FROZEN_DECISION_SPEC_V1.md",
    protocol_doc_path: Path = REPO_ROOT / "docs" / "FROZEN_EVALUATION_PROTOCOL_V1.md",
    blind_path: Path = REPO_ROOT / "datasets" / "blind_external_template.json",
) -> dict[str, Any]:
    if not spec_path.exists():
        raise FileNotFoundError(f"Missing decision spec: {spec_path}")
    if not protocol_doc_path.exists():
        raise FileNotFoundError(f"Missing evaluation protocol doc: {protocol_doc_path}")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset = generate_frozen_splits(seed=GENERATOR_SEED)
    audit = audit_frozen_dataset(dataset)
    if not audit["integrity_passed"]:
        raise AssertionError(f"Frozen dataset audit failed: {audit}")

    blind_info = load_blind_external_cases(blind_path)

    # Store inputs and oracle labels strictly in separate JSONL files
    files = {
        "dev_inputs": dataset_dir / "dev_inputs.jsonl",
        "dev_oracle_labels": dataset_dir / "dev_oracle_labels.jsonl",
        "test_inputs": dataset_dir / "test_inputs.jsonl",
        "test_oracle_labels": dataset_dir / "test_oracle_labels.jsonl",
        "test_update_inputs": dataset_dir / "test_update_paths.jsonl",
        "test_update_oracle_labels": dataset_dir / "test_update_oracle_labels.jsonl",
    }
    _write_jsonl(files["dev_inputs"], dataset["dev_inputs"])
    _write_jsonl(files["dev_oracle_labels"], dataset["dev_labels"])
    _write_jsonl(files["test_inputs"], dataset["test_inputs"])
    _write_jsonl(files["test_oracle_labels"], dataset["test_labels"])
    _write_jsonl(files["test_update_inputs"], dataset["update_inputs"])
    _write_jsonl(files["test_update_oracle_labels"], dataset["update_labels"])

    hashes = {
        "decision_spec_v1": {
            "path": str(spec_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "sha256": _sha256_file(spec_path),
        },
        "evaluation_protocol_v1": {
            "path": str(protocol_doc_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "sha256": _sha256_file(protocol_doc_path),
        },
    }
    for name, fpath in files.items():
        hashes[name] = {
            "path": str(fpath).replace("\\", "/"),
            "sha256": _sha256_file(fpath),
        }

    budget_plan = build_groq_8k_tpm_budget_plan(
        test_count=audit["test_count"],
        update_count=audit["update_paths_count"],
    )

    manifest = {
        "phase": "Phase 5 (Frozen Dataset, Spec, Baselines, Stats, Budget; ZERO test evaluation run)",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit_sha": _git_head(),
        "spec_version": SPEC_VERSION,
        "generator_version": GENERATOR_VERSION,
        "generator_seed": GENERATOR_SEED,
        "sweep_seeds": list(SWEEP_SEEDS),
        "authorized_live_model": {
            "provider": "groq",
            "model_id": "openai/gpt-oss-20b",
            "api_key_env_var": "GROQ_API_KEY",
            "temperature": 0.0,
            "min_call_interval_sec": 5.0,
        },
        "test_set_executed_in_phase5": False,
        "hashes": hashes,
        "dataset_audit": audit,
        "blind_teammate_cases": {
            "status": blind_info["status"],
            "count": blind_info["count"],
            "target_count": blind_info["target_count"],
            "path": blind_info["path"],
            "sha256": blind_info["sha256"],
            "notes": blind_info["notes"],
        },
        "primary_baselines": list(PRIMARY_BASELINES_V1),
        "deprecated_historical_reference": DEPRECATED_HISTORICAL_ARM_NOTE,
        "targeted_ablations": list(TARGETED_ABLATIONS_V1),
        "statistical_protocol": frozen_statistical_protocol(),
        "live_budget_plan": budget_plan,
    }

    manifest_path = results_dir / "frozen_evaluation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    md = [
        "# Phase 5 Frozen Dataset Integrity & Protocol Audit",
        "",
        f"- **Generated (UTC):** `{manifest['generated_utc']}`",
        f"- **Commit SHA:** `{manifest['code_commit_sha']}`",
        f"- **Spec Version:** `{SPEC_VERSION}` (`sha256={hashes['decision_spec_v1']['sha256']}`)",
        f"- **Protocol Doc:** `protocol-v1` (`sha256={hashes['evaluation_protocol_v1']['sha256']}`)",
        f"- **Test Set Evaluated in Phase 5:** `{manifest['test_set_executed_in_phase5']}` (Phase 6 only)",
        f"- **Blind Teammate Cases:** `{blind_info['status']}` (`count={blind_info['count']}/{blind_info['target_count']}`)",
        "",
        "## Dataset Split Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total Distinct Claims (`dev` + `test`) | {audit['total_claims']} |",
        f"| Unique Claim IDs | {audit['unique_claim_ids']} |",
        f"| Unique Canonical Fact Hashes | {audit['unique_fact_hashes']} |",
        f"| `dev` Split Claims / Families | {audit['dev_count']} / {audit['dev_family_count']} |",
        f"| `test` Split Claims / Families | {audit['test_count']} / {audit['test_family_count']} |",
        f"| Overlapping Families (`dev` ∩ `test`) | {audit['overlapping_family_count']} |",
        f"| Overlapping Fact Hashes (`dev` ∩ `test`) | {audit['overlapping_fact_hash_count']} |",
        f"| Overlapping Non-APPROVE Bool Signatures | {audit['overlapping_non_approve_boolean_signatures']} |",
        f"| `test` Update Paths (`v1 -> v2`) | {audit['update_paths_count']} |",
        f"| Independent Spec vs. Rule-Oracle Parity | `{audit['all_labels_cross_checked']}` |",
        f"| Integrity Audit Passed | `{audit['integrity_passed']}` |",
        "",
        "## Template Family Breakdown",
        "",
        "| Split | Template Family | Count | Unique Fact Hashes | APPROVE | REJECT | REVIEW |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in audit["family_table"]:
        md.append(
            f"| `{row['split']}` | `{row['family_id']}` | {row['count']} | {row['unique_fact_hashes']} | {row['APPROVE']} | {row['REJECT']} | {row['REVIEW']} |"
        )
    md.extend([
        "",
        "## Frozen File SHA-256 Manifest",
        "",
        "| Artifact | SHA-256 |",
        "| --- | --- |",
    ])
    for name, info in hashes.items():
        md.append(f"| `{name}` (`{info['path']}`) | `{info['sha256']}` |")
    (results_dir / "dataset_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze Phase 5 dataset splits, oracle labels, and protocol manifest.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/benchmark_v1"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/phase5"))
    args = parser.parse_args()
    manifest = freeze_phase5(dataset_dir=args.dataset_dir, results_dir=args.results_dir)
    print(json.dumps({
        "spec_version": manifest["spec_version"],
        "spec_sha256": manifest["hashes"]["decision_spec_v1"]["sha256"],
        "total_claims": manifest["dataset_audit"]["total_claims"],
        "test_count": manifest["dataset_audit"]["test_count"],
        "overlapping_families": manifest["dataset_audit"]["overlapping_family_count"],
        "overlapping_fact_hashes": manifest["dataset_audit"]["overlapping_fact_hash_count"],
        "blind_status": manifest["blind_teammate_cases"]["status"],
        "test_set_executed_in_phase5": manifest["test_set_executed_in_phase5"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
