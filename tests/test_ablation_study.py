"""Behavioral tests for the offline ablation study logic and metric calculations."""
from dataclasses import replace
from pathlib import Path
import pytest

from cert_recovery.adjudication.ablation import (
    ANNOTATION_STATUS,
    HUMAN_REVIEWED,
    METHOD_REGISTRY,
    SPEC_VERSION,
    calculate_metrics,
    calculate_paired_disagreements,
    evaluate_arms,
    extended_scenarios,
    historical_cases,
    run_ablation_study,
    save_ablation_results,
)
from cert_recovery.adjudication.domain import InsuranceClaim, ROLES, rule_decision
from cert_recovery.adjudication.workflow import InsuranceAdjudication


def test_frozen_scenario_spec_and_oracle_consistency():
    assert SPEC_VERSION == "ablation-v1"
    assert ANNOTATION_STATUS == "SYNTHETIC_PROVISIONAL"
    assert HUMAN_REVIEWED is False

    scenarios = extended_scenarios()
    assert len(scenarios) == 10
    for s in scenarios:
        claim = s["claim"]
        local = [rule_decision(r, claim.facts(r)) for r in ROLES]
        oracle = "REJECT" if "REJECT" in local else "REVIEW" if "REVIEW" in local else "APPROVE"
        assert s["label"] == oracle, f"Mismatch in scenario {s['case_id']}"


def test_historical_cases_consistency():
    cases = historical_cases()
    assert len(cases) == 5
    expected_labels = ["APPROVE", "REJECT", "REVIEW", "REVIEW", "REVIEW"]
    assert [c["label"] for c in cases] == expected_labels


def test_nine_arms_evaluated_on_identical_execution():
    claim = InsuranceClaim()
    workflow = InsuranceAdjudication(claim, faults={"risk_agent": "inflated_confidence"})
    data = workflow.export()

    arms = evaluate_arms(claim, data)
    assert set(arms.keys()) == {"A", "B", "C", "D", "E", "F", "G", "H", "I"}
    for k, v in arms.items():
        assert v in {"APPROVE", "REJECT", "REVIEW"}


def test_multi_fault_distinguishes_detection_and_gate():
    claim = InsuranceClaim()
    # 3 corrupted agents voting REJECT with 0.999 confidence
    faults = {r + "_agent": "inflated_confidence" for r in ROLES[:3]}
    workflow = InsuranceAdjudication(claim, faults=faults)
    data = workflow.export()

    arms = evaluate_arms(claim, data)
    # A (Majority) and C (Weighted unvalidated) fall for the corruption and REJECT
    assert arms["A"] == "REJECT"
    assert arms["C"] == "REJECT"
    # B (Majority + Gate) catches conflict with rule, forces REVIEW
    assert arms["B"] == "REVIEW"
    # D (Weighted validated) filters out corrupted agents, leaving valid compliance agent -> APPROVE
    assert arms["D"] == "APPROVE"
    # E (Full prototype) validates and gate passes -> APPROVE
    assert arms["E"] == "APPROVE"
    # F (Pure rules oracle) evaluates clean claim facts -> APPROVE
    assert arms["F"] == "APPROVE"
    # G (Majority + Validated + Gate): Validated output has compliance agent voting APPROVE -> gate passes -> APPROVE
    assert arms["G"] == "APPROVE"
    # H (Majority + Validated, no gate): Validated output has compliance agent voting APPROVE -> APPROVE
    assert arms["H"] == "APPROVE"
    # I (Weighting + Gate, no validation): unvalidated votes REJECT -> conflicts with oracle APPROVE -> REVIEW
    assert arms["I"] == "REVIEW"


def test_metric_calculations_separates_decided_and_review():
    labels = ["APPROVE", "REJECT", "REVIEW", "REVIEW", "APPROVE"]
    predictions = ["APPROVE", "REVIEW", "REVIEW", "APPROVE", "REJECT"]

    metrics = calculate_metrics(predictions, labels)
    assert metrics["total_cases"] == 5
    # Matches: index 0 (APPROVE==APPROVE), index 2 (REVIEW==REVIEW) -> 2/5 (40%)
    assert metrics["reported_correct"] == 2
    assert metrics["reported_accuracy"] == 0.4

    # Decided cases: index 0, 3, 4 (total 3 decided)
    assert metrics["decided_count"] == 3
    assert metrics["coverage"] == 3 / 5
    # Correct decided: index 0 (1/3)
    assert metrics["decided_accuracy"] == 1 / 3

    # Strict accuracy: only index 0 is non-review correct -> 1/5
    assert metrics["strict_correct"] == 1
    assert metrics["strict_accuracy"] == 0.2

    # Unsafe approval: index 3 (pred APPROVE, label REVIEW)
    assert metrics["unsafe_approvals"] == 1
    # Incorrect rejection: index 4 (pred REJECT, label APPROVE)
    assert metrics["incorrect_rejections"] == 1
    # False review: index 1 (pred REVIEW, label REJECT)
    assert metrics["false_reviews"] == 1


def test_paired_disagreements_quantifies_transitions():
    p1 = ["APPROVE", "APPROVE", "REVIEW", "REJECT"]
    p2 = ["APPROVE", "REVIEW", "REVIEW", "APPROVE"]

    paired = calculate_paired_disagreements(p1, p2)
    assert paired["total_comparisons"] == 4
    assert paired["disagreement_count"] == 2
    assert paired["disagreement_rate"] == 0.5
    assert paired["transitions"] == {"APPROVE->REVIEW": 1, "REJECT->APPROVE": 1}


def test_ablation_prohibits_overwriting_existing_results(tmp_path):
    target = tmp_path / "test_results.json"
    data = {"spec_version": "test"}
    save_ablation_results(data, target)
    assert target.exists()

    with pytest.raises(FileExistsError):
        save_ablation_results(data, target)


def test_bounded_ablation_run_completes_offline():
    # Test a small bounded run: 1 seed, 2 rates, historical suite (10 trials total)
    res = run_ablation_study(cases_set="historical", rates=(0.0, 0.2), seeds=(42,))
    assert res["total_trials"] == 10
    expected_arms = {"A", "B", "C", "D", "E", "F", "G", "H", "I"}
    assert set(res["overall_by_method"].keys()) == expected_arms
    assert len(res["rate_summaries"]) == 2
    for entry in res["rate_summaries"]:
        assert entry["total_trials"] == 5
        assert entry["total_agent_slots"] == 20
        assert set(entry["methods"].keys()) == expected_arms
    assert "single_factor_effects" in res
    assert "gate_effect" in res["single_factor_effects"]
    assert "validation_effect" in res["single_factor_effects"]
    assert "weighting_effect" in res["single_factor_effects"]
