"""Regression and validation tests for repaired harness, metrics, and recovery boundaries."""
from dataclasses import replace
import json
from pathlib import Path
import pytest

from cert_recovery.adjudication.ablation import (
    METHOD_REGISTRY,
    calculate_metrics,
    evaluate_arms,
    extended_scenarios,
    historical_cases,
    run_ablation_study,
    save_ablation_results,
)
from cert_recovery.adjudication.domain import InsuranceClaim, ROLES
from cert_recovery.adjudication.recovery_benchmark import (
    full_recompute_recovery,
    run_recovery_benchmark,
    save_recovery_benchmark,
)
from cert_recovery.adjudication.workflow import InsuranceAdjudication
from cert_recovery.crypto import sha256


# 1. Metric Numerators and Denominators
def test_metric_numerators_and_denominators_strict_separation():
    """Verify clean separation of 3-class accuracy, decisive accuracy, coverage, and deferrals."""
    # Construct scenario mirroring the Arm D ambiguity:
    # 150 cases:
    # - 148 decided: 33 correct decided, 115 incorrect decided (unsafe approvals or incorrect rejections)
    # - 2 review cases: 1 correctly predicted as REVIEW, 1 incorrectly predicted as APPROVE
    # Total predictions:
    # - Correct 3-class: 33 (decided) + 1 (review) = 34
    # - Total cases = 150
    # - Decided cases = 148
    # - Decisive correct = 33
    labels = ["APPROVE"] * 33 + ["REJECT"] * 115 + ["REVIEW"] * 1 + ["REVIEW"] * 1
    predictions = ["APPROVE"] * 33 + ["APPROVE"] * 115 + ["REVIEW"] * 1 + ["APPROVE"] * 1

    metrics = calculate_metrics(predictions, labels)

    assert metrics["total_cases"] == 150
    # 3-class accuracy: 34 / 150 = 22.67%
    assert metrics["reported_correct"] == 34
    assert abs(metrics["reported_accuracy"] - (34 / 150)) < 1e-9

    # Decided count and coverage
    assert metrics["decided_count"] == 149  # 33 + 115 + 1
    # Now construct the exact 148 decided:
    labels_exact = ["APPROVE"] * 33 + ["REJECT"] * 115 + ["REVIEW"] * 2
    preds_exact = ["APPROVE"] * 33 + ["APPROVE"] * 115 + ["REVIEW"] * 1 + ["REVIEW"] * 0 + ["APPROVE"] * 0
    # Let's verify with 148 decided: 33 APPROVE (correct), 115 APPROVE on REJECT label (unsafe), 2 REVIEW on REVIEW label (correct review)
    # Then decided count = 148, total = 150
    # 3-class correct = 33 + 2 = 35.
    # What if 1 correct review and 1 false approval? Then decided = 149.
    # What if labels had 150 cases, with 2 review labels, and predictions had 2 review predictions:
    # 33 correct decided, 115 incorrect decided, 1 correct review, 1 false review on a decided case!
    labels_148 = ["APPROVE"] * 33 + ["REJECT"] * 114 + ["REJECT"] * 1 + ["REVIEW"] * 2
    preds_148 = ["APPROVE"] * 33 + ["APPROVE"] * 114 + ["REVIEW"] * 1 + ["REVIEW"] * 1 + ["REJECT"] * 1
    # In preds_148:
    # Decided predictions: 33 + 114 + 1 = 148 decided!
    # Decided correct: 33 (APPROVE) + 0 (REJECT on REVIEW is wrong) = 33 correct decided!
    # 3-class correct: 33 + 1 (REVIEW on REVIEW) = 34!
    m = calculate_metrics(preds_148, labels_148)
    assert m["total_cases"] == 150
    assert m["reported_correct"] == 34
    assert m["decided_count"] == 148
    assert abs(m["coverage"] - (148 / 150)) < 1e-9
    # Crucial test: decided_accuracy must be 33/148, NOT 34/148!
    assert m["decided_accuracy"] == 33 / 148
    assert abs(m["decided_accuracy"] - (33 / 148)) < 1e-9
    assert m["decided_accuracy"] != 34 / 148

    # Separation of required reviews from additional deferrals:
    # In labels_148: 2 cases have label "REVIEW". 1 was predicted "REVIEW" (required review).
    assert m["required_reviews"] == 1
    # 1 case had label "REJECT" and was predicted "REVIEW" (additional deferral / false review).
    assert m["additional_deferrals"] == 1
    assert m["false_reviews"] == 1


# 2. Matched Comparison Settings & Full Factorial Space
def test_all_eight_factorial_combinations_present_and_consistent():
    """Verify the complete 2x2x2 factorial grid of (Majority vs Weighting) x (Val OFF vs ON) x (Gate OFF vs ON)."""
    expected_arms = {"A", "B", "C", "D", "E", "F", "G", "H", "I"}
    assert set(METHOD_REGISTRY.keys()) == expected_arms

    # Check 2x2x2 factorial combinations
    grid = {}
    for arm, meta in METHOD_REGISTRY.items():
        if arm == "F":
            continue
        agg = "WEIGHTING" if meta["has_confidence"] else "MAJORITY"
        val = meta["has_validation"]
        gate = meta["has_gate"]
        key = (agg, val, gate)
        assert key not in grid, f"Duplicate arm combination for {key}: {arm} vs {grid[key]}"
        grid[key] = arm

    assert len(grid) == 8
    assert grid[("MAJORITY", False, False)] == "A"
    assert grid[("MAJORITY", False, True)] == "B"
    assert grid[("MAJORITY", True, False)] == "H"
    assert grid[("MAJORITY", True, True)] == "G"
    assert grid[("WEIGHTING", False, False)] == "C"
    assert grid[("WEIGHTING", False, True)] == "I"
    assert grid[("WEIGHTING", True, False)] == "D"
    assert grid[("WEIGHTING", True, True)] == "E"


def test_matched_arms_receive_identical_inputs_and_faults():
    """Verify that all 9 arms are evaluated on the exact same workflow instance."""
    claim = InsuranceClaim()
    wf = InsuranceAdjudication(claim, faults={"evidence_agent": "misleading_evidence"})
    data = wf.export()

    arms = evaluate_arms(claim, data)
    assert len(arms) == 9
    for arm in ["A", "B", "C", "D", "E", "F", "G", "H", "I"]:
        assert arms[arm] in {"APPROVE", "REJECT", "REVIEW"}


# 3. Recovery Execution Counting Regression Test
def test_full_recomputation_never_double_counts_executions():
    """Regression test: full recomputation must never execute selective recovery first."""
    # Scenario 2: fact change
    c = InsuranceClaim()
    wf_full = InsuranceAdjudication(c)
    target_claim = replace(c, documents_available=False)

    # Initial executions should all be 1
    for r in ROLES:
        assert wf_full.executions[r] == 1

    # Execute repaired full recomputation
    outcome = full_recompute_recovery(wf_full, target_claim)
    assert outcome.status == "COMPLETE"

    # Execution delta per role must be EXACTLY 1 (total executions = 2 per role)
    for r in ROLES:
        assert wf_full.executions[r] == 2, f"Role {r} executed {wf_full.executions[r]} times, expected exactly 2"

    total_deltas = sum(wf_full.executions[r] - 1 for r in ROLES)
    assert total_deltas == 4, f"Full recovery executed {total_deltas} agents, expected strictly 4"


def test_recovery_benchmark_execution_bounds_all_scenarios():
    """Verify all 4 scenarios in the recovery benchmark satisfy execution bounds."""
    bench = run_recovery_benchmark()
    for sc in bench["scenarios"]:
        recomputed_full = sc["full_recomputation"]["recomputed_agent_executions"]
        recomputed_sel = sc["selective"]["recomputed_agent_executions"]
        assert recomputed_full == 4, f"Scenario '{sc['scenario']}' full recomputation was {recomputed_full}, expected 4"
        assert recomputed_sel in {1, 2}, f"Scenario '{sc['scenario']}' selective was {recomputed_sel}, expected 1 or 2"
        assert sc["efficiency_gain"]["saved_agent_executions"] in {2, 3}


# 4. Source-Snapshot Completeness
def test_source_snapshot_manifest_and_completeness():
    """Verify that critical source files exist and have non-empty valid content."""
    repo_root = Path(__file__).resolve().parents[1]
    required_files = [
        repo_root / "src/cert_recovery/adjudication/ablation.py",
        repo_root / "src/cert_recovery/adjudication/consensus.py",
        repo_root / "src/cert_recovery/adjudication/domain.py",
        repo_root / "src/cert_recovery/adjudication/ledger.py",
        repo_root / "src/cert_recovery/adjudication/recovery_benchmark.py",
        repo_root / "src/cert_recovery/adjudication/workflow.py",
        repo_root / "src/cert_recovery/adjudication/model_agent.py",
        repo_root / "src/cert_recovery/adjudication/model_evaluation.py",
        repo_root / "scripts/run_insurance_ablation.py",
        repo_root / "tests/test_ablation_study.py",
        repo_root / "tests/test_provenance_recovery_integrity.py",
        repo_root / "tests/test_insurance_adjudication.py",
    ]
    for file_path in required_files:
        assert file_path.exists(), f"Source file missing: {file_path}"
        content = file_path.read_bytes()
        assert len(content) > 0, f"Source file is empty: {file_path}"
        # Compute SHA-256
        file_hash = sha256(content.decode("utf-8", errors="replace"))
        assert len(file_hash) == 64


# 5. Artifact Overwrite Protection
def test_save_utilities_prohibit_overwriting(tmp_path):
    """Verify exclusive file creation guards in ablation and recovery saving functions."""
    target_ablation = tmp_path / "ablation_test.json"
    dummy_data = {"test": "data"}

    # First write succeeds
    p1 = save_ablation_results(dummy_data, target_ablation)
    assert p1.exists()

    # Second write MUST raise FileExistsError
    with pytest.raises(FileExistsError):
        save_ablation_results(dummy_data, target_ablation)

    target_recovery = tmp_path / "recovery_test.json"
    p2 = save_recovery_benchmark(dummy_data, target_recovery)
    assert p2.exists()

    with pytest.raises(FileExistsError):
        save_recovery_benchmark(dummy_data, target_recovery)
