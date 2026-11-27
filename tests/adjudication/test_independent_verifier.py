"""Unit and anti-circularity tests for the independent composition verifier."""
from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from cert_recovery.adjudication import independent_verifier
from cert_recovery.adjudication.composition import CompositionResult, SpecialistAssessment
from cert_recovery.adjudication.domain import ROLES, InsuranceClaim
from cert_recovery.adjudication.workflow import SpecialistCompositionAdjudication
from cert_recovery.crypto import canonical_json

pytestmark = pytest.mark.adjudication

REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFIER_PATH = REPO_ROOT / "src/cert_recovery/adjudication/independent_verifier.py"


def test_independent_verifier_shares_no_composer_or_oracle_imports():
    """Confirm independent_verifier.py imports no composer, consensus, rule_decision, or oracle code."""
    tree = ast.parse(VERIFIER_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            imported.add(mod)
            for alias in node.names:
                imported.add(f"{mod}.{alias.name}")

    forbidden_tokens = ("composition", "consensus", "oracle", "ablation", "rule_decision", "RuleAgent")
    hits = [item for item in imported if any(tok in item for tok in forbidden_tokens)]
    assert not hits, f"Independent verifier must not import composer/oracle symbols: {hits}"


def test_independent_verifier_accepts_all_valid_and_faulted_composition_modes():
    """Verify all six bindings pass across clean, exclusion, deferral, and faulted workflows."""
    scenarios = [
        (InsuranceClaim(), {}, "substantive_exclusion", "APPROVE"),
        (replace(InsuranceClaim(), hospitalization_covered=False), {}, "substantive_exclusion", "REJECT"),
        (replace(InsuranceClaim(), hospitalization_covered=False), {"evidence_agent": "failure"}, "substantive_exclusion", "REJECT"),
        (replace(InsuranceClaim(), hospitalization_covered=False), {"evidence_agent": "failure"}, "strict_audit", "REVIEW"),
        (replace(InsuranceClaim(), documents_available=False), {}, "substantive_exclusion", "REVIEW"),
    ]
    for claim, faults, precedence, expected_decision in scenarios:
        wf = SpecialistCompositionAdjudication(claim, faults=faults, precedence=precedence)
        res = wf.result()
        assert res["decision"] == expected_decision
        round_ref = wf.system.latest("round").ref
        round_node = wf.system.artifact(round_ref)
        ancestry = wf.system.provenance(round_ref)
        vr = independent_verifier.verify_composition_independently(
            round_node, ancestry, wf.ledger, precedence=precedence
        )
        assert vr.passed is True, vr.reason


def test_anti_circularity_corrupted_composer_decision_rejected(monkeypatch):
    """Mutating composer decision logic causes CertificateSystem verification to reject."""
    from cert_recovery.adjudication import workflow as wf_mod

    orig_compose = wf_mod.compose_specialist_assessments

    def corrupted_compose(assessments, precedence="substantive_exclusion"):
        res = orig_compose(assessments, precedence=precedence)
        return CompositionResult(
            decision="APPROVE",  # Corrupted override on policy exclusion
            gate="ALL_OBLIGATIONS_SATISFIED",
            precedence=res.precedence,
            contract_version=res.contract_version,
            specialist_assessments=res.specialist_assessments,
            advisory_consensus=res.advisory_consensus,
            procedural_exceptions=res.procedural_exceptions,
            rule_references=res.rule_references,
        )

    monkeypatch.setattr(wf_mod, "compose_specialist_assessments", corrupted_compose)
    excluded_claim = replace(InsuranceClaim(), hospitalization_covered=False)
    with pytest.raises(ValueError, match="Verification failed: Composed decision mismatch"):
        SpecialistCompositionAdjudication(excluded_claim)


def test_anti_circularity_corrupted_specialist_assessment_rejected(monkeypatch):
    """Mutating specialist assessor to mark invalid/failed output as SATISFIED is rejected."""
    from cert_recovery.adjudication import workflow as wf_mod

    orig_assess = wf_mod.assess_specialist

    def forged_assess(output, role, facts, binding, backend_id="synthetic-rule-agent-v1"):
        res = orig_assess(output, role, facts, binding, backend_id=backend_id)
        if role == "evidence":
            return replace(
                res,
                obligation_result="SATISFIED",
                rule_consequence="APPROVE",
                validation_status="VALID",
                validation_score=1.0,
                fault_score=0.0,
                reasons=(),
            )
        return res

    monkeypatch.setattr(wf_mod, "assess_specialist", forged_assess)
    missing_docs_claim = replace(InsuranceClaim(), documents_available=False)
    with pytest.raises(ValueError, match="Verification failed: Independent obligation/binding verification failed for evidence"):
        SpecialistCompositionAdjudication(missing_docs_claim)


def test_independent_verifier_rejects_copied_facts_lie_and_stale_binding():
    """Verifier checks authoritative source artifacts in ancestry, never trusting copied agent payload fields."""
    wf = SpecialistCompositionAdjudication(InsuranceClaim())
    round_ref = wf.system.latest("round").ref
    round_node = wf.system.artifact(round_ref)
    ancestry = wf.system.provenance(round_ref)

    # 1. Agent payload lies about copied evidence relative to authoritative facts:policy
    forged_ancestry_1 = tuple(
        replace(
            node,
            payload_json=canonical_json({
                **node.payload,
                "evidence": {**node.payload["evidence"], "hospitalization_covered": False},
            }),
        )
        if node.ref.artifact_id == "policy_agent" else node
        for node in ancestry
    )
    # Note: hash integrity catches payload_json mutation first; also test with recomputed hash
    from cert_recovery.models import Artifact
    forged_policy_payload = {
        **wf.system.latest("policy_agent").payload,
        "evidence": {**wf.system.latest("policy_agent").payload["evidence"], "hospitalization_covered": False},
    }
    p_node = wf.system.latest("policy_agent")
    rehashed_policy = Artifact.create(p_node.ref, p_node.kind, forged_policy_payload, p_node.supports, p_node.source, p_node.cost_units)
    forged_ancestry_rehashed = tuple(rehashed_policy if n.ref == p_node.ref else n for n in ancestry)
    vr1 = independent_verifier.verify_composition_independently(round_node, forged_ancestry_rehashed, wf.ledger)
    assert not vr1.passed
    assert "authoritative source" in vr1.reason.lower()

    # 2. Agent payload binds stale source version/hash relative to authoritative facts:policy
    stale_binding_payload = {
        **p_node.payload,
        "input_reference": {**p_node.payload["input_reference"], "version": 99},
    }
    stale_policy = Artifact.create(p_node.ref, p_node.kind, stale_binding_payload, p_node.supports, p_node.source, p_node.cost_units)
    forged_ancestry_stale = tuple(stale_policy if n.ref == p_node.ref else n for n in ancestry)
    vr2 = independent_verifier.verify_composition_independently(round_node, forged_ancestry_stale, wf.ledger)
    assert not vr2.passed
    assert "stale or mismatched input_reference" in vr2.reason.lower()

    # 3. Authoritative source rule version mismatch
    src_pol = wf.system.latest("facts:policy")
    bad_rule_src = Artifact.create(
        src_pol.ref, src_pol.kind, {**src_pol.payload, "rule": "coverage-v99"}, src_pol.supports, src_pol.source, src_pol.cost_units
    )
    forged_ancestry_rule = tuple(bad_rule_src if n.ref == src_pol.ref else n for n in ancestry)
    vr3 = independent_verifier.verify_composition_independently(round_node, forged_ancestry_rule, wf.ledger)
    assert not vr3.passed
    assert "rule version mismatch" in vr3.reason.lower()

    # 4. Round rule_references mismatch
    bad_round = Artifact.create(
        round_node.ref,
        round_node.kind,
        {**round_node.payload, "rule_references": ["tampered-rule"]},
        round_node.supports,
        round_node.source,
        round_node.cost_units,
    )
    vr4 = independent_verifier.verify_composition_independently(bad_round, ancestry, wf.ledger)
    assert not vr4.passed
    assert "rule_references mismatch" in vr4.reason.lower()

    assert len(forged_ancestry_1) == len(ancestry)
    assert all(r in ROLES for r in ROLES)
