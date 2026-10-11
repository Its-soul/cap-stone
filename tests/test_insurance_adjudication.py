"""Synthetic behavioral tests; no model, network, or research checkpoints."""
import copy
from dataclasses import replace

import pytest

from cert_recovery.adjudication.consensus import ConfidenceWeightedConsensus, Vote, assess, majority
from cert_recovery.adjudication.domain import InsuranceClaim, PROMPTS, ROLES, RuleAgent
from cert_recovery.adjudication.experiments import comparison, run_experiments
from cert_recovery.adjudication.ledger import EventDAG
from cert_recovery.adjudication.workflow import InsuranceAdjudication
from cert_recovery.crypto import sha256
from cert_recovery.models import CertificateStatus
from cert_recovery.provenance import AuditJournal


def test_four_distinct_roles_and_structured_outputs():
    workflow = InsuranceAdjudication(InsuranceClaim())
    data = workflow.export()
    assert data["final"]["decision"] == "APPROVE"
    assert workflow.executions == dict.fromkeys(ROLES, 1)
    assert len(set(PROMPTS.values())) == 4
    assert len({tuple(a["evidence"]) for a in data["agents"]}) == 4
    assert len({tuple(a["rule_references"]) for a in data["agents"]}) == 4
    for agent in data["agents"]:
        raw = agent["raw_output"]
        assert 0 <= raw["confidence"] <= 1
        assert raw["input_reference"] == agent["input_reference"]
        assert raw["reasoning_summary"]
        assert agent["assessment"]["trust_status"] == "VALIDATED"
        assert agent["provenance_reference"]


def test_confidence_changes_outcome_against_majority():
    mechanism = ConfidenceWeightedConsensus()
    votes = [Vote("a", "APPROVE", .2), Vote("b", "APPROVE", .2), Vote("c", "REJECT", .9)]
    assert majority([{"decision": v.decision} for v in votes]) == "APPROVE"
    result = mechanism.aggregate(votes)
    assert result["decision"] == "REJECT"
    assert result["totals"]["APPROVE"] == .4
    assert mechanism.aggregate([replace(votes[0], confidence=.8), *votes[1:]])["decision"] == "APPROVE"
    assert mechanism.aggregate(list(reversed(votes))) == result


def test_ties_empty_duplicate_and_validation_weights():
    mechanism = ConfidenceWeightedConsensus()
    assert mechanism.aggregate([])["decision"] == "REVIEW"
    assert mechanism.aggregate([Vote("a", "APPROVE", .5), Vote("b", "REJECT", .5)])["abstained"]
    assert mechanism.aggregate([Vote("a", "APPROVE", 0)])["abstained"]
    with pytest.raises(ValueError, match="Duplicate"):
        mechanism.aggregate([Vote("a", "APPROVE", .5)] * 2)
    result = mechanism.aggregate([Vote("a", "APPROVE", .4), Vote("b", "REJECT", 1., fault_score=1.)])
    assert result["decision"] == "APPROVE" and result["weights"]["b"] == 0
    assert mechanism.aggregate([Vote("a", "APPROVE", .8, reliability=.5, validation_score=.5, fault_score=.5)])["weights"]["a"] == .1


@pytest.mark.parametrize("value", [-.1, 1.1, float("nan"), float("inf"), True, "0.9"])
def test_invalid_confidence_rejected(value):
    with pytest.raises(ValueError):
        Vote("a", "APPROVE", value)


def test_calibration_hook_and_entropy_are_explicit():
    class FixtureCalibrator:
        def transform(self, agent_id, confidence):
            return .1 if agent_id == "a" else confidence
    votes = [Vote("a", "APPROVE", .9), Vote("b", "REJECT", .5)]
    assert ConfidenceWeightedConsensus().aggregate(votes)["decision"] == "APPROVE"
    result = ConfidenceWeightedConsensus(FixtureCalibrator()).aggregate(votes)
    assert result["decision"] == "REJECT"
    assert result["calibration_transform_applied"] and result["vote_entropy"] > 0
    assert result["calibration_status"] == "external_transform_unverified"


@pytest.mark.parametrize("mode,status", [("incorrect", "INCORRECT"), ("inflated_confidence", "SUSPICIOUS"),
    ("misleading_evidence", "SUSPICIOUS"), ("contradictory", "SUSPICIOUS"),
    ("malformed", "CONFIRMED_FAILURE"), ("failure", "CONFIRMED_FAILURE")])
def test_fault_modes_excluded_and_recorded(mode, status):
    data = InsuranceAdjudication(InsuranceClaim(), faults={"risk_agent": mode}).export()
    risk = next(a for a in data["agents"] if a["role"] == "risk")
    assert risk["assessment"]["trust_status"] == status
    assert data["round"]["consensus"]["weights"]["risk_agent"] == 0
    assert data["final"]["decision"] == "APPROVE"
    assert EventDAG.verify(data["events"], data["dag_checkpoint"])


def test_stale_binding_identity_and_unknown_rule_rejected():
    facts, binding = InsuranceClaim().facts("policy"), {"version": 1}
    raw = RuleAgent().execute("policy", PROMPTS["policy"], facts, binding)
    assert assess(raw, "policy", facts, {"version": 2}).trust_status == "SUSPICIOUS"
    assert assess({**raw, "agent_id": "risk_agent"}, "policy", facts, binding).trust_status == "CONFIRMED_FAILURE"
    assert assess({**raw, "rule_references": []}, "policy", facts, binding).trust_status == "SUSPICIOUS"


@pytest.mark.parametrize("raw", [None, [], "APPROVE", {"confidence": float("nan")}, {"confidence": object()}])
def test_untrusted_backend_is_contained(raw):
    class BrokenBackend:
        def execute(self, *args):
            return raw
    data = InsuranceAdjudication(InsuranceClaim(), backends={"risk": BrokenBackend()}).export()
    risk = next(a for a in data["agents"] if a["role"] == "risk")
    assert risk["assessment"]["trust_status"] == "CONFIRMED_FAILURE"
    assert data["final"]["decision"] == "APPROVE"


@pytest.mark.parametrize("change", [{"documents_available": False}, {"duplicate_claim": True},
    {"identity_verified": False}, {"hospitalization_covered": False}, {"amount_inr": 150_000}])
def test_mandatory_rules_prevent_unsafe_majority_approval(change):
    data = InsuranceAdjudication(replace(InsuranceClaim(), **change)).export()
    assert data["round"]["baseline"] == "APPROVE"
    assert data["final"]["decision"] == "REVIEW"
    assert data["final"]["gate"] == "REVIEW_REQUIRED"
    # The prototype abstains on exclusions instead of pretending to fully adjudicate rejection.


def test_all_agents_failed_abstains():
    data = InsuranceAdjudication(InsuranceClaim(), faults={r + "_agent": "failure" for r in ROLES}).export()
    assert data["round"]["consensus"]["total_weight"] == 0
    assert data["final"]["decision"] == "REVIEW"


def test_rule_dag_and_event_trace_bind_sources_votes_and_certificate():
    workflow = InsuranceAdjudication(InsuranceClaim())
    data = workflow.export()
    trace = workflow.ledger.trace(data["final"]["provenance_reference"])
    types = {e["event_type"] for e in trace}
    assert {"input_received", "evidence_access", "agent_execution", "agent_output", "validation",
            "fault_assessment", "vote", "consensus", "final_decision"} <= types
    event = next(e for e in trace if e["event_type"] == "consensus")
    assert len(event["parents"]) == 4
    nodes = {n["ref"]["artifact_id"]: n for n in data["causal_rule_nodes"]}
    assert nodes["policy_agent"]["supports"] == [{"artifact_id": "facts:policy", "version": 1}]
    assert len(nodes["round"]["supports"]) == 4
    assert nodes["final"]["supports"] == [{"artifact_id": "round-certificate", "version": 1}]
    assert AuditJournal.verify(data["audit_chain"], workflow.system.checkpoint())


@pytest.mark.parametrize("mutation", ["payload", "parent", "timestamp", "delete", "reorder", "truncate", "hash"])
def test_cryptographic_tamper_detection(mutation):
    data = InsuranceAdjudication(InsuranceClaim()).export()
    records = copy.deepcopy(data["events"])
    assert EventDAG.verify(records, data["dag_checkpoint"])
    if mutation == "payload": records[0]["payload"]["forged"] = True
    elif mutation == "parent": records[-1]["parents"] = []
    elif mutation == "timestamp": records[0]["timestamp"] = "forged"
    elif mutation == "delete": del records[2]
    elif mutation == "reorder": records[1], records[2] = records[2], records[1]
    elif mutation == "truncate": records.pop()
    else: records[0]["event_hash"] = "0" * 64
    assert not EventDAG.verify(records, data["dag_checkpoint"])


def test_canonical_dag_and_external_checkpoint_boundary():
    first, second = EventDAG(clock=lambda: "fixed"), EventDAG(clock=lambda: "fixed")
    assert first.append("input", {"a": 1, "b": 2}) == second.append("input", {"b": 2, "a": 1})
    with pytest.raises(ValueError): first.append("bad", {}, ("unknown",))
    pinned = first.checkpoint()
    records = first.export()
    records[0]["payload"]["a"] = 99
    records[0]["payload_hash"] = sha256(records[0]["payload"])
    records[0]["event_hash"] = sha256({k: v for k, v in records[0].items() if k != "event_hash"})
    assert EventDAG.verify(records)  # A fully rehashed unanchored history cannot authenticate itself.
    assert not EventDAG.verify(records, pinned)


def test_selective_recovery_preserves_valid_agents_and_certificate_history():
    workflow = InsuranceAdjudication(InsuranceClaim(), faults={"risk_agent": "failure"})
    old = {r: workflow.system.latest(r + "_agent") for r in ROLES}
    certificate = workflow.system.latest("round-certificate").ref
    outcome = workflow.recover_agent("risk_agent")
    assert outcome.status == "COMPLETE"
    assert {r.artifact_id for r in outcome.restored} == {"risk_agent", "round", "round-certificate", "final"}
    for role in ROLES:
        if role != "risk":
            assert workflow.system.latest(role + "_agent") == old[role]
    assert workflow.executions == {"policy": 1, "evidence": 1, "risk": 2, "compliance": 1}
    assert workflow.system.certificate_status(certificate) == CertificateStatus.SUPERSEDED
    assert workflow.result()["decision"] == "APPROVE"
    assert "recovery_started" in {r["event_type"] for r in workflow.ledger.trace(workflow.result()["provenance_reference"])}
    assert EventDAG.verify(workflow.ledger.export(), workflow.ledger.checkpoint())
    assert AuditJournal.verify(workflow.system.audit_export(), workflow.system.checkpoint())


def test_fact_change_rebuilds_only_dependent_role_and_new_decision():
    workflow = InsuranceAdjudication(InsuranceClaim())
    old_final = workflow.system.latest("final").ref
    outcome = workflow.update_claim(replace(InsuranceClaim(), duplicate_claim=True))
    assert outcome.status == "COMPLETE"
    assert workflow.executions == {"policy": 1, "evidence": 1, "risk": 2, "compliance": 1}
    assert workflow.result()["decision"] == "REVIEW"
    with pytest.raises(ValueError): workflow.system.commit_decision(old_final)
    assert workflow.update_claim(replace(InsuranceClaim(), duplicate_claim=True)) is None


def test_persistent_fault_is_not_silently_repaired():
    workflow = InsuranceAdjudication(InsuranceClaim(), faults={"risk_agent": "failure"})
    workflow.recover_agent("risk_agent", clear_simulated_fault=False)
    assert workflow.system.latest("risk_agent").payload["assessment"]["trust_status"] == "CONFIRMED_FAILURE"
    assert workflow.result()["consensus"]["weights"]["risk_agent"] == 0


def test_verifier_rejects_forged_round_and_assessment():
    workflow = InsuranceAdjudication(InsuranceClaim())
    round_node = workflow.system.latest("round")
    from cert_recovery.crypto import canonical_json
    forged = replace(round_node, payload_json=canonical_json({**round_node.payload, "decision": "REJECT"}))
    assert not workflow._verify_round(forged, ()).passed
    forged_event = replace(round_node, payload_json=canonical_json({**round_node.payload, "provenance_reference": "unknown"}))
    assert not workflow._verify_round(forged_event, ()).passed


def test_failed_round_reverification_blocks_commit_and_keeps_unaffected_agents(monkeypatch):
    workflow = InsuranceAdjudication(InsuranceClaim())
    original_round = workflow._round
    def forged_round(supports):
        return {**original_round(supports), "decision": "REJECT"}
    monkeypatch.setattr(workflow, "_round", forged_round)
    outcome = workflow.recover_agent("risk_agent")
    assert outcome.status == "PARTIAL"
    assert "round-certificate" in dict(outcome.blocked)
    with pytest.raises(ValueError): workflow.result()
    assert workflow.executions == {"policy": 1, "evidence": 1, "risk": 2, "compliance": 1}


def test_backend_cannot_mutate_trusted_facts():
    class MutatingBackend:
        def execute(self, role, prompt, facts, binding):
            facts["duplicate_claim"] = True
            binding["version"] = 999
            return RuleAgent().execute(role, prompt, facts, binding)
    workflow = InsuranceAdjudication(InsuranceClaim(), backends={"risk": MutatingBackend()})
    assert workflow.system.latest("facts:risk").payload["facts"]["duplicate_claim"] is False
    assert workflow.system.latest("risk_agent").payload["assessment"]["trust_status"] == "SUSPICIOUS"


def test_baseline_comparison_is_measured_from_same_outputs():
    results = comparison()
    assert results["all_correct"]["baseline"] == results["all_correct"]["proposed"] == "APPROVE"
    assert results["multiple_faults"]["baseline"] == "REJECT"
    assert results["multiple_faults"]["proposed"] == "APPROVE"


def test_seeded_experiments_reconstruct_metrics_and_honest_failures():
    result = run_experiments(trials=5, seed=42)
    assert result == run_experiments(trials=5, seed=42)
    assert len(result["trials"]) == 25
    for summary in result["summaries"]:
        rows = [r for r in result["trials"] if r["requested_rate"] == summary["requested_rate"]]
        assert summary["proposed_accuracy"] == sum(r["proposed"] == r["label"] for r in rows) / 5
        assert summary["realized_rate"] == sum(r["affected_agents"] for r in rows) / 20
        assert all(r["recovery_executions"] == r["affected_agents"] for r in rows)
    assert result["summaries"][0]["proposed_accuracy"] == .8  # Policy rejection remains REVIEW.
    assert result["summaries"][0]["detection_recall"] is None
    assert result["summaries"][0]["recovery_success_rate"] is None


@pytest.mark.parametrize("kwargs", [{"faults": {"unknown": "failure"}}, {"faults": {"risk_agent": "unknown"}},
                                   {"backends": {"unknown": RuleAgent()}}])
def test_unknown_configuration_rejected(kwargs):
    with pytest.raises(ValueError): InsuranceAdjudication(InsuranceClaim(), **kwargs)
