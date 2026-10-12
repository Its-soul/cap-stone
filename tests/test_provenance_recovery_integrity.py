"""Behavioral verification of provenance, selective recovery, and cryptographic integrity."""
import copy
from dataclasses import replace
import pytest

from cert_recovery.adjudication.consensus import assess
from cert_recovery.adjudication.domain import InsuranceClaim, ROLES, RuleAgent, PROMPTS, RULES
from cert_recovery.adjudication.ledger import EventDAG
from cert_recovery.adjudication.recovery_benchmark import run_recovery_benchmark, save_recovery_benchmark
from cert_recovery.adjudication.workflow import InsuranceAdjudication
from cert_recovery.crypto import sha256
from cert_recovery.models import CertificateStatus


def test_changed_evidence_and_stale_agent_outputs_detected():
    """Agent outputs bound to old input versions/hashes are caught and excluded."""
    claim = InsuranceClaim()
    facts_v1 = claim.facts("policy")
    binding_v1 = {"artifact_id": "facts:policy", "version": 1, "content_hash": "hash_v1"}

    raw = RuleAgent().execute("policy", PROMPTS["policy"], facts_v1, binding_v1)
    # 1. Matches exact binding
    assert assess(raw, "policy", facts_v1, binding_v1).trust_status == "VALIDATED"

    # 2. Stale binding: facts updated to v2, but agent presents output with v1 binding
    binding_v2 = {"artifact_id": "facts:policy", "version": 2, "content_hash": "hash_v2"}
    stale_assessment = assess(raw, "policy", facts_v1, binding_v2)
    assert stale_assessment.trust_status == "SUSPICIOUS"
    assert "stale_or_unbound_input" in stale_assessment.reasons
    assert stale_assessment.validation_score == 0.0
    assert stale_assessment.fault_score == 1.0


def test_failed_agent_recovery_and_certificate_lifecycle():
    """Failed agent invalidates certificates; recovery restores valid decision."""
    workflow = InsuranceAdjudication(InsuranceClaim(), faults={"risk_agent": "failure"})
    old_cert_ref = workflow.system.latest("round-certificate").ref
    assert workflow.system.certificate_status(old_cert_ref) == CertificateStatus.VALID
    # Decision cannot commit if risk agent output was a failure and corrupted
    # Risk agent output is failure, assigned weight 0.0, but system can still decide if consensus passes.
    # Now recover the agent:
    outcome = workflow.recover_agent("risk_agent")
    assert outcome.status == "COMPLETE"
    assert "risk_agent" in [r.artifact_id for r in outcome.restored]
    assert "round-certificate" in [r.artifact_id for r in outcome.restored]

    # Verify old certificate was SUPERSEDED, not silently reused
    assert workflow.system.certificate_status(old_cert_ref) == CertificateStatus.SUPERSEDED
    new_cert_ref = workflow.system.latest("round-certificate").ref
    assert new_cert_ref.version == old_cert_ref.version + 1
    assert workflow.system.certificate_status(new_cert_ref) == CertificateStatus.VALID
    assert workflow.result()["decision"] == "APPROVE"


def test_repeated_changes_before_recovery():
    """Multiple successive changes to dependencies before recovery plan in topological order."""
    claim = InsuranceClaim()
    workflow = InsuranceAdjudication(claim)

    # 1. Change policy facts
    s_pol = workflow.system.latest("facts:policy")
    workflow.system.change_dependency("facts:policy", {**s_pol.payload, "facts": {**s_pol.payload["facts"], "amount_inr": 80_000}})

    # 2. Change evidence facts
    s_evi = workflow.system.latest("facts:evidence")
    workflow.system.change_dependency("facts:evidence", {**s_evi.payload, "facts": {**s_evi.payload["facts"], "documents_available": False}})

    # 3. Change policy facts AGAIN before recovering
    s_pol2 = workflow.system.latest("facts:policy")
    workflow.system.change_dependency("facts:policy", {**s_pol2.payload, "facts": {**s_pol2.payload["facts"], "amount_inr": 90_000}})

    # Now execute pending recovery
    outcome = workflow._recover()
    assert outcome.status == "COMPLETE"
    restored_ids = [r.artifact_id for r in outcome.restored]
    assert "policy_agent" in restored_ids
    assert "evidence_agent" in restored_ids
    # Risk and compliance must not be rebuilt
    assert "risk_agent" not in restored_ids
    assert "compliance_agent" not in restored_ids
    assert workflow.executions == {"policy": 2, "evidence": 2, "risk": 1, "compliance": 1}
    assert workflow.result()["decision"] == "REVIEW"


def test_multiple_affected_branches_and_preservation_of_independent_valid_branches():
    """Changing facts in two roles rebuilds those roles and preserves unaffected branches."""
    claim = InsuranceClaim()
    workflow = InsuranceAdjudication(claim)

    initial_risk = workflow.system.latest("risk_agent")
    initial_comp = workflow.system.latest("compliance_agent")

    new_claim = replace(claim, hospitalization_covered=False, invoice_matches=False)
    outcome = workflow.update_claim(new_claim)

    assert outcome.status == "COMPLETE"
    assert workflow.executions["policy"] == 2
    assert workflow.executions["evidence"] == 2
    assert workflow.executions["risk"] == 1
    assert workflow.executions["compliance"] == 1

    # Assert exact version, hash, and reference preservation for independent branches
    current_risk = workflow.system.latest("risk_agent")
    current_comp = workflow.system.latest("compliance_agent")

    assert current_risk.ref == initial_risk.ref
    assert current_risk.content_hash == initial_risk.content_hash
    assert current_comp.ref == initial_comp.ref
    assert current_comp.content_hash == initial_comp.content_hash


def test_missing_or_failed_rebuilders_block_commit():
    """A broken rebuilder stops recovery and strictly blocks decision commit."""
    claim = InsuranceClaim()
    workflow = InsuranceAdjudication(claim)

    # Inject broken rebuilder for risk_agent
    def broken_recipe(*args):
        raise RuntimeError("Rebuilder failed explicitly")

    workflow.system._recipes["risk_agent"] = broken_recipe
    report = workflow.system.invalidate_artifact(workflow.system.latest("risk_agent").ref, "invalidation")
    plan = workflow.system.plan_recovery(report)
    outcome = workflow.system.execute_recovery(plan)

    assert outcome.status == "BLOCKED"
    blocked_dict = dict(outcome.blocked)
    assert "risk_agent" in blocked_dict
    assert "Rebuilder failed explicitly" in blocked_dict["risk_agent"]
    assert "round" in blocked_dict
    assert "round-certificate" in blocked_dict

    # Decision commit must be blocked
    assert not workflow.system.is_valid(workflow.system.latest("final").ref)
    with pytest.raises(ValueError, match="Decision cannot commit with stale/invalid certificates"):
        workflow.result()


def test_recovery_budget_exhaustion_halts_recovery():
    """Insufficient recovery budget halts execution, leaving artifacts blocked."""
    claim = InsuranceClaim()
    workflow = InsuranceAdjudication(claim)

    report = workflow.system.invalidate_artifact(workflow.system.latest("risk_agent").ref, "test budget")
    # Budget of 1.5 units: risk_agent costs 1.0, but round costs 1.0, so round exceeds budget
    plan = workflow.system.plan_recovery(report, budget_units=1.5)
    outcome = workflow.system.execute_recovery(plan)

    assert outcome.status == "PARTIAL"
    assert "risk_agent" in [r.artifact_id for r in outcome.restored]
    blocked_dict = dict(outcome.blocked)
    assert "round" in blocked_dict
    assert blocked_dict["round"] == "budget insufficient"
    assert not workflow.system.is_valid(workflow.system.latest("final").ref)


def test_cryptographic_tamper_detection_against_retained_trusted_anchor():
    """All structural and content mutations fail verification against an external anchor."""
    workflow = InsuranceAdjudication(InsuranceClaim())
    data = workflow.export()
    original_records = copy.deepcopy(data["events"])
    trusted_checkpoint = data["dag_checkpoint"]

    # 0. Clean baseline passes
    assert EventDAG.verify(original_records, trusted_checkpoint)

    # 1. Payload change
    rec = copy.deepcopy(original_records)
    rec[2]["payload"]["tampered_key"] = "forged"
    assert not EventDAG.verify(rec, trusted_checkpoint)

    # 2. Parent change (altering parent hash)
    rec = copy.deepcopy(original_records)
    rec[-1]["parents"] = ["forged_parent_hash"]
    assert not EventDAG.verify(rec, trusted_checkpoint)

    # 3. Missing parents (parent list empty on non-root event)
    rec = copy.deepcopy(original_records)
    rec[-1]["parents"] = []
    assert not EventDAG.verify(rec, trusted_checkpoint)

    # 4. Deletion of an intermediate event
    rec = copy.deepcopy(original_records)
    del rec[3]
    assert not EventDAG.verify(rec, trusted_checkpoint)

    # 5. Reordering of events
    rec = copy.deepcopy(original_records)
    rec[1], rec[2] = rec[2], rec[1]
    assert not EventDAG.verify(rec, trusted_checkpoint)

    # 6. Truncation (popping the tip event)
    rec = copy.deepcopy(original_records)
    rec.pop()
    assert not EventDAG.verify(rec, trusted_checkpoint)

    # 7. Counterexample: Unanchored rehashed history cannot authenticate itself
    # If attacker rewrites the history and recalculates SHA256 hashes internally:
    unanchored_dag = EventDAG(clock=lambda: "fixed")
    unanchored_dag.append("input", {"data": "authentic_record"})
    trusted_anchor = unanchored_dag.checkpoint()

    rehashed_records = unanchored_dag.export()
    rehashed_records[0]["payload"]["data"] = "tampered_record"
    rehashed_records[0]["payload_hash"] = sha256(rehashed_records[0]["payload"])
    rehashed_records[0]["event_hash"] = sha256({k: v for k, v in rehashed_records[0].items() if k != "event_hash"})

    # Without the external trusted anchor, unanchored self-verification passes:
    assert EventDAG.verify(rehashed_records)
    # With the retained external anchor, the forgery is immediately detected and rejected:
    assert not EventDAG.verify(rehashed_records, trusted_anchor)


def test_event_dag_and_artifact_graph_cross_consistency():
    """Event DAG and Artifact Graph remain synchronized through recovery."""
    workflow = InsuranceAdjudication(InsuranceClaim(), faults={"risk_agent": "failure"})
    workflow.recover_agent("risk_agent")

    final_artifact = workflow.system.latest("final")
    provenance_hash = final_artifact.payload["provenance_reference"]

    # Trace through EventDAG from final decision
    trace = workflow.ledger.trace(provenance_hash)
    trace_types = [e["event_type"] for e in trace]

    assert "final_decision" in trace_types
    assert "recovery_started" in trace_types
    assert "consensus" in trace_types
    assert "agent_execution" in trace_types
    assert "evidence_access" in trace_types

    # Event DAG verification against current checkpoint passes
    assert EventDAG.verify(workflow.ledger.export(), workflow.ledger.checkpoint())


def test_recovery_benchmark_runs_and_saves_exclusive(tmp_path):
    """Recovery benchmark executes across all 4 scenarios and produces valid records."""
    bench = run_recovery_benchmark()
    assert len(bench["scenarios"]) == 4
    for sc in bench["scenarios"]:
        assert sc["selective"]["status"] == "COMPLETE"
        assert sc["full_recomputation"]["status"] == "COMPLETE"
        assert sc["efficiency_gain"]["saved_agent_executions"] > 0
        assert sc["efficiency_gain"]["saved_cost_units"] > 0

    target = tmp_path / "recovery_benchmark.json"
    save_recovery_benchmark(bench, target)
    assert target.exists()
    with pytest.raises(FileExistsError):
        save_recovery_benchmark(bench, target)
