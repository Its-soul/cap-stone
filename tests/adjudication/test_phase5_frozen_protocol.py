"""Offline unit tests verifying Phase 5 frozen specification, splits, oracle isolation, and stats."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cert_recovery.oracle.frozen_v1 import (
    GENERATOR_SEED,
    SPEC_VERSION,
    audit_frozen_dataset,
    derive_spec_v1_labels,
    generate_frozen_splits,
    load_blind_external_cases,
)
from cert_recovery.oracle.protocol_v1 import (
    build_groq_8k_tpm_budget_plan,
    compute_phase6_metrics,
    exact_mcnemar_test,
    paired_bootstrap_ci,
    wilson_ci,
)
from scripts.freeze_phase5_protocol import freeze_phase5

pytestmark = pytest.mark.adjudication

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_frozen_dataset_has_over_120_unique_claims_and_zero_split_overlap():
    """Confirm >=100 unique test claims, >=120 total claims, and 0 family/fact overlap across splits."""
    ds1 = generate_frozen_splits(seed=GENERATOR_SEED)
    ds2 = generate_frozen_splits(seed=GENERATOR_SEED)
    assert ds1 == ds2

    audit = audit_frozen_dataset(ds1)
    assert audit["integrity_passed"] is True
    assert audit["total_claims"] >= 120
    assert audit["test_count"] >= 100
    assert audit["unique_claim_ids"] == audit["total_claims"]
    assert audit["unique_fact_hashes"] == audit["total_claims"]
    assert audit["dev_family_count"] >= 4
    assert audit["test_family_count"] >= 8
    assert audit["overlapping_family_count"] == 0
    assert audit["overlapping_fact_hash_count"] == 0
    assert audit["overlapping_non_approve_boolean_signatures"] == 0
    assert audit["all_labels_cross_checked"] is True
    assert audit["update_paths_count"] == 32
    assert set(audit["update_kind_distribution"]) == {
        "narrow_single_role",
        "broad_multi_role",
        "no_change_edit",
        "same_label_value_edit",
    }
    assert set(audit["test_class_distribution"]) == {"APPROVE", "REJECT", "REVIEW"}


def test_independent_spec_v1_evaluator_boundaries():
    """Verify exact equality, 1-INR limit breach, and precedence rules from FROZEN_DECISION_SPEC_V1.md."""
    base = {
        "claim_id": "spec-boundary-test",
        "amount_inr": 100_000,
        "coverage_limit_inr": 100_000,
        "hospitalization_covered": True,
        "waiting_period_satisfied": True,
        "documents_available": True,
        "invoice_matches": True,
        "duplicate_claim": False,
        "provider_flagged": False,
        "identity_verified": True,
        "consent_available": True,
    }
    res_eq = derive_spec_v1_labels(base)
    assert res_eq["spec_version"] == SPEC_VERSION
    assert res_eq["obligations"]["policy"] == "SAT"
    assert res_eq["label"] == "APPROVE"
    assert "exact_coverage_limit_equality" in res_eq["edge_flags"]

    res_over = derive_spec_v1_labels({**base, "amount_inr": 100_001})
    assert res_over["obligations"]["policy"] == "FAIL"
    assert res_over["label"] == "REJECT"
    assert "one_rupee_over_coverage_limit" in res_over["edge_flags"]

    res_mixed = derive_spec_v1_labels({**base, "hospitalization_covered": False, "documents_available": False})
    assert res_mixed["label"] == "REJECT"
    assert "mixed_exclusion_and_procedural_failure" in res_mixed["edge_flags"]


def test_blind_teammate_loader_honest_pending_and_fake_rejection(tmp_path: Path):
    """Template stays honest PENDING_TEAMMATE_SUBMISSION; fake or non-blind submissions are rejected."""
    tpl = REPO_ROOT / "datasets" / "blind_external_template.json"
    info = load_blind_external_cases(tpl)
    assert info["status"] == "PENDING_TEAMMATE_SUBMISSION"
    assert info["count"] == 0

    fake_file = tmp_path / "fake_blind.json"
    fake_file.write_text(
        json.dumps({
            "schema_version": "blind-external-v1",
            "cases": [{
                "case_id": "blind-1",
                "author": "synthetic",
                "authored_blind_to_system_outputs": True,
                "review_status": "BLIND_HUMAN_AUTHORED",
                "claim": {
                    "claim_id": "blind-1",
                    "amount_inr": 1000,
                    "coverage_limit_inr": 2000,
                    "hospitalization_covered": True,
                    "waiting_period_satisfied": True,
                    "documents_available": True,
                    "invoice_matches": True,
                    "duplicate_claim": False,
                    "provider_flagged": False,
                    "identity_verified": True,
                    "consent_available": True,
                },
            }],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid/non-human author"):
        load_blind_external_cases(fake_file)

    non_blind = tmp_path / "non_blind.json"
    payload = json.loads(fake_file.read_text(encoding="utf-8"))
    payload["cases"][0]["author"] = "Teammate-A"
    payload["cases"][0]["authored_blind_to_system_outputs"] = False
    non_blind.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="authored_blind_to_system_outputs == true"):
        load_blind_external_cases(non_blind)


def test_phase5_freeze_inputs_labels_separated_and_hashes_reproducible(tmp_path: Path):
    """Verify frozen JSONL inputs omit oracle labels, separate oracle JSONL stores them, and hashes match."""
    d1 = tmp_path / "run1_ds"
    r1 = tmp_path / "run1_res"
    d2 = tmp_path / "run2_ds"
    r2 = tmp_path / "run2_res"

    m1 = freeze_phase5(dataset_dir=d1, results_dir=r1)
    m2 = freeze_phase5(dataset_dir=d2, results_dir=r2)

    assert m1["test_set_executed_in_phase5"] is False
    assert m1["deprecated_historical_reference"]["status"] == "DROPPED_FROM_PRIMARY_COMPARISONS"

    for key in (
        "decision_spec_v1",
        "evaluation_protocol_v1",
        "dev_inputs",
        "dev_oracle_labels",
        "test_inputs",
        "test_oracle_labels",
        "test_update_inputs",
        "test_update_oracle_labels",
    ):
        assert m1["hashes"][key]["sha256"] == m2["hashes"][key]["sha256"]

    # Confirm test_inputs.jsonl carries zero oracle labels
    first_input = json.loads((d1 / "test_inputs.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "label" not in first_input
    assert "obligations" not in first_input

    # Verify disk SHA-256 matches manifest
    disk_sha = hashlib.sha256((d1 / "test_inputs.jsonl").read_bytes()).hexdigest()
    assert disk_sha == m1["hashes"]["test_inputs"]["sha256"]


def test_statistical_protocol_metrics_wilson_bootstrap_and_mcnemar():
    """Verify exact denominators, Wilson CI, family-clustered bootstrap CI, McNemar test, and Groq budget."""
    y_true = ["APPROVE", "REJECT", "REVIEW", "REJECT"]
    preds = ["APPROVE", "APPROVE", "REVIEW", "REJECT"]
    m = compute_phase6_metrics(preds, y_true)
    # 3 non-approve cases total, 1 unsafe approval -> 1/3; 2 non-reject cases -> 0/2 false rejects
    assert m["total_non_approve_cases"] == 3
    assert m["unsafe_approval_rate"] == pytest.approx(1.0 / 3.0)
    assert m["false_reject_rate"] == pytest.approx(0.0)
    assert m["coverage"] == pytest.approx(0.75)
    assert m["review_rate"] == pytest.approx(0.25)
    assert m["selective_accuracy"] == pytest.approx(2.0 / 3.0)
    assert m["exact_match_accuracy"] == pytest.approx(0.75)

    w = wilson_ci(0, 100)
    assert w["low"] == 0.0
    assert 0.03 < w["high"] < 0.04

    mc = exact_mcnemar_test([True, True, True, False], [False, False, False, False])
    assert mc["n_a_only"] == 3
    assert mc["n_b_only"] == 0
    assert mc["exact_two_sided_p"] == pytest.approx(0.25)

    boot = paired_bootstrap_ci(
        [1, 1, 0, 1],
        [0, 1, 0, 0],
        clusters=["f1", "f1", "f2", "f2"],
        n_resamples=200,
    )
    assert boot["clustered"] is True
    assert boot["ci95_low"] <= boot["observed_diff"] <= boot["ci95_high"]

    budget = build_groq_8k_tpm_budget_plan(test_count=104, update_count=32)
    assert budget["planning_assumptions"]["projected_tpm"] < budget["planning_assumptions"]["groq_free_tier_tpm_limit"]
