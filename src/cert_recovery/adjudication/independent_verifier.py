"""Independent verifier for specialist-role composition certificates.

This module intentionally shares NO decision, validation, or composition logic
with `composition.py`, `consensus.py`, or `domain.rule_decision`, and imports
nothing from `cert_recovery.oracle`. It verifies a `round` artifact directly
from authoritative stored source artifacts in `ancestry` and the `EventDAG`
ledger against the frozen `composition-v1` specification.
"""
from __future__ import annotations

from math import isfinite
from typing import Any

from ..crypto import canonical_json
from ..models import Artifact, ArtifactKind, VerificationResult
from .domain import DECISIONS, ROLES
from .ledger import EventDAG

EXPECTED_CONTRACT_VERSION = "composition-v1"
EXPECTED_ROLE_CONTRACTS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "policy": (
        "coverage-v1",
        "policy_eligibility",
        ("amount_inr", "coverage_limit_inr", "hospitalization_covered", "waiting_period_satisfied"),
    ),
    "evidence": (
        "documents-v1",
        "evidence_integrity",
        ("documents_available", "invoice_matches"),
    ),
    "risk": (
        "fraud-screen-v1",
        "screening_integrity",
        ("duplicate_claim", "provider_flagged"),
    ),
    "compliance": (
        "identity-consent-v1",
        "regulatory_controls",
        ("identity_verified", "consent_available"),
    ),
}
EXPECTED_PERMITTED_CONSEQUENCES: dict[str, tuple[str, ...]] = {
    "policy": ("APPROVE", "REJECT"),
    "evidence": ("APPROVE", "REVIEW"),
    "risk": ("APPROVE", "REVIEW"),
    "compliance": ("APPROVE", "REVIEW"),
}
SUPPORTED_PRECEDENCE = ("substantive_exclusion", "strict_audit")


def _evaluate_spec_rule(role: str, facts: dict[str, Any]) -> str:
    """Evaluate governing domain predicate directly from the specification."""
    if role == "policy":
        amount = facts["amount_inr"]
        limit = facts["coverage_limit_inr"]
        if type(amount) is not int or amount <= 0 or type(limit) is not int or limit <= 0:
            raise ValueError("Policy amounts must be positive integers")
        covered = facts["hospitalization_covered"]
        waiting = facts["waiting_period_satisfied"]
        if type(covered) is not bool or type(waiting) is not bool:
            raise ValueError("Policy flags must be booleans")
        return "APPROVE" if (covered and waiting and amount <= limit) else "REJECT"

    if role == "evidence":
        docs = facts["documents_available"]
        inv = facts["invoice_matches"]
        if type(docs) is not bool or type(inv) is not bool:
            raise ValueError("Evidence flags must be booleans")
        return "APPROVE" if (docs and inv) else "REVIEW"

    if role == "risk":
        dup = facts["duplicate_claim"]
        flagged = facts["provider_flagged"]
        if type(dup) is not bool or type(flagged) is not bool:
            raise ValueError("Risk flags must be booleans")
        return "APPROVE" if not (dup or flagged) else "REVIEW"

    if role == "compliance":
        ident = facts["identity_verified"]
        consent = facts["consent_available"]
        if type(ident) is not bool or type(consent) is not bool:
            raise ValueError("Compliance flags must be booleans")
        return "APPROVE" if (ident and consent) else "REVIEW"

    raise ValueError(f"Unknown role: {role}")


def _derive_obligation_from_authoritative_source(
    raw_output: Any,
    role: str,
    authoritative_facts: dict[str, Any],
    authoritative_binding: dict[str, Any],
) -> tuple[str, str, str, str]:
    """Independently derive (short_status, obligation_result, rule_consequence, validation_status)."""
    expected_rule = EXPECTED_ROLE_CONTRACTS[role][0]

    if not isinstance(raw_output, dict) or "execution_error" in raw_output:
        return "UNK", "UNCERTAIN", "REVIEW", "FAILED"

    conf = raw_output.get("confidence")
    if type(conf) not in (int, float) or not isfinite(conf) or not (0.0 <= conf <= 1.0):
        return "UNK", "UNCERTAIN", "REVIEW", "FAILED"

    decision = raw_output.get("decision")
    if (
        decision not in DECISIONS
        or raw_output.get("agent_id") != f"{role}_agent"
        or raw_output.get("role") != role
        or not isinstance(raw_output.get("reasoning_summary"), str)
        or not raw_output.get("reasoning_summary")
        or not isinstance(raw_output.get("evidence"), dict)
        or not isinstance(raw_output.get("rule_references"), list)
    ):
        return "UNK", "UNCERTAIN", "REVIEW", "FAILED"

    if (
        decision not in EXPECTED_PERMITTED_CONSEQUENCES[role]
        or raw_output.get("input_reference") != authoritative_binding
        or raw_output.get("evidence") != authoritative_facts
        or raw_output.get("rule_references") != [expected_rule]
        or decision != _evaluate_spec_rule(role, authoritative_facts)
    ):
        return "UNK", "UNCERTAIN", "REVIEW", "INVALID"

    if role == "policy":
        if decision == "APPROVE":
            return "SAT", "SATISFIED", "APPROVE", "VALID"
        return "FAIL", "FAILED", "REJECT", "VALID"

    if decision == "APPROVE":
        return "SAT", "SATISFIED", "APPROVE", "VALID"
    return "FAIL", "FAILED", "REVIEW", "VALID"


def _expected_composition_decision(
    obligations: dict[str, str],
    precedence: str,
) -> tuple[str, str]:
    """Evaluate the frozen composition-v1 truth table directly from (SAT/FAIL/UNK)."""
    if precedence == "substantive_exclusion":
        if obligations["policy"] == "UNK":
            return "REVIEW", "POLICY_UNCERTAIN"
        if obligations["policy"] == "FAIL":
            return "REJECT", "SUBSTANTIVE_EXCLUSION"
        if any(obligations[r] in ("FAIL", "UNK") for r in ("evidence", "risk", "compliance")):
            return "REVIEW", "PROCEDURAL_DEFERRAL"
        if all(obligations[r] == "SAT" for r in ROLES):
            return "APPROVE", "ALL_OBLIGATIONS_SATISFIED"
        return "REVIEW", "FALLTHROUGH_REVIEW"

    if precedence == "strict_audit":
        if any(obligations[r] == "UNK" for r in ROLES):
            return "REVIEW", "AUDIT_UNCERTAINTY"
        if obligations["policy"] == "FAIL":
            return "REJECT", "SUBSTANTIVE_EXCLUSION"
        if any(obligations[r] == "FAIL" for r in ("evidence", "risk", "compliance")):
            return "REVIEW", "PROCEDURAL_DEFERRAL"
        if all(obligations[r] == "SAT" for r in ROLES):
            return "APPROVE", "ALL_OBLIGATIONS_SATISFIED"
        return "REVIEW", "FALLTHROUGH_REVIEW"

    raise ValueError(f"Unsupported precedence mode: {precedence}")


def verify_composition_independently(
    subject: Artifact,
    ancestry: tuple[Artifact, ...],
    ledger: Any,
    precedence: str = "substantive_exclusion",
) -> VerificationResult:
    """Independently verify a composition round artifact against authoritative sources.

    Verifies all six bindings:
    1. Exact artifact/evidence versions (subject -> agents -> facts:<role> in ancestry)
    2. Authoritative payload hashes (artifact integrity_valid, input bindings, EventDAG)
    3. Support references (declared DAG edges and EventDAG causal parent links)
    4. Rule specification version (role rules, composition-v1 contract, rule_references)
    5. Validated obligation statuses (SAT/FAIL/UNK derived from authoritative facts)
    6. Permitted final decision and gate under the configured precedence rules
    """
    if precedence not in SUPPORTED_PRECEDENCE:
        return VerificationResult(False, f"Unsupported precedence mode: {precedence}")

    try:
        if subject.kind != ArtifactKind.CLAIM or not subject.integrity_valid():
            return VerificationResult(False, "Subject round artifact integrity check failed")

        by_ref = {node.ref: node for node in ancestry}
        for node in ancestry:
            if not node.integrity_valid():
                return VerificationResult(False, f"Ancestry artifact hash mismatch: {node.ref}")

        if len(subject.supports) != len(ROLES):
            return VerificationResult(False, "Round must declare exactly four specialist supports")

        agent_nodes: list[Artifact] = []
        for expected_role, support_ref in zip(ROLES, subject.supports):
            if support_ref.artifact_id != f"{expected_role}_agent":
                return VerificationResult(False, f"Support order/role mismatch: {support_ref}")
            agent_node = by_ref.get(support_ref)
            if agent_node is None or agent_node.kind != ArtifactKind.EVIDENCE:
                return VerificationResult(False, f"Missing or invalid agent artifact in ancestry: {support_ref}")
            agent_nodes.append(agent_node)

        subject_payload = subject.payload
        if subject_payload.get("contract_version") != EXPECTED_CONTRACT_VERSION:
            return VerificationResult(False, "Composition contract version mismatch")
        if subject_payload.get("precedence") != precedence:
            return VerificationResult(False, "Composition precedence mismatch")

        expected_round_rules = [EXPECTED_ROLE_CONTRACTS[r][0] for r in ROLES] + [EXPECTED_CONTRACT_VERSION]
        if subject_payload.get("rule_references") != expected_round_rules:
            return VerificationResult(False, "Composition rule_references mismatch")

        # Verify EventDAG integrity before looking up provenance commitments
        exported_events = ledger.export()
        if not EventDAG.verify(exported_events, ledger.checkpoint()):
            return VerificationResult(False, "EventDAG integrity verification failed")
        events_by_hash = {e["event_hash"]: e for e in exported_events}

        assessments_in_payload = subject_payload.get("specialist_assessments")
        if not isinstance(assessments_in_payload, dict) or set(assessments_in_payload) != set(ROLES):
            return VerificationResult(False, "Malformed specialist_assessments map")

        short_obligations: dict[str, str] = {}
        agent_prov_refs: list[str] = []

        for role, agent_node in zip(ROLES, agent_nodes):
            expected_rule, expected_obligation_name, expected_fact_keys = EXPECTED_ROLE_CONTRACTS[role]

            # 1 & 3: Support reference must bind exactly to facts:<role> DEPENDENCY in ancestry
            if len(agent_node.supports) != 1 or agent_node.supports[0].artifact_id != f"facts:{role}":
                return VerificationResult(False, f"Agent support binding mismatch for {role}")
            source_node = by_ref.get(agent_node.supports[0])
            if source_node is None or source_node.kind != ArtifactKind.DEPENDENCY:
                return VerificationResult(False, f"Authoritative source dependency missing for {role}")

            source_payload = source_node.payload
            if source_payload.get("rule") != expected_rule:
                return VerificationResult(False, f"Authoritative source rule version mismatch for {role}")

            authoritative_facts = source_payload.get("facts")
            if not isinstance(authoritative_facts, dict) or set(authoritative_facts) != set(expected_fact_keys):
                return VerificationResult(False, f"Authoritative source facts schema invalid for {role}")

            authoritative_binding = {
                "artifact_id": source_node.ref.artifact_id,
                "version": source_node.ref.version,
                "content_hash": source_node.content_hash,
            }

            # 2: Do NOT trust copied facts/bindings in agent payload without checking authoritative source
            agent_payload = agent_node.payload
            if agent_payload.get("role") != role or agent_payload.get("agent_id") != f"{role}_agent":
                return VerificationResult(False, f"Agent payload role/id mismatch for {role}")
            if agent_payload.get("rule_references") != [expected_rule]:
                return VerificationResult(False, f"Agent payload rule_references mismatch for {role}")
            if agent_payload.get("input_reference") != authoritative_binding:
                return VerificationResult(False, f"Stale or mismatched input_reference hash/version for {role}")
            if agent_payload.get("evidence") != authoritative_facts:
                return VerificationResult(False, f"Agent payload evidence differs from authoritative source for {role}")

            # 5: Derive obligation status independently from raw_output + authoritative facts/binding
            short_status, exp_ob_res, exp_conseq, exp_val_status = _derive_obligation_from_authoritative_source(
                agent_payload.get("raw_output"),
                role,
                authoritative_facts,
                authoritative_binding,
            )
            short_obligations[role] = short_status

            actual_spec = assessments_in_payload.get(role)
            if not isinstance(actual_spec, dict):
                return VerificationResult(False, f"Missing specialist assessment entry for {role}")
            if (
                actual_spec.get("role") != role
                or actual_spec.get("contract_version") != expected_rule
                or actual_spec.get("obligation_name") != expected_obligation_name
                or actual_spec.get("obligation_result") != exp_ob_res
                or actual_spec.get("rule_consequence") != exp_conseq
                or actual_spec.get("validation_status") != exp_val_status
                or actual_spec.get("input_binding") != authoritative_binding
            ):
                return VerificationResult(False, f"Independent obligation/binding verification failed for {role}")

            # Verify expected validation/fault scores
            exp_val_score = 1.0 if exp_val_status == "VALID" else 0.0
            exp_fault_score = 0.0 if exp_val_status == "VALID" else 1.0
            if actual_spec.get("validation_score") != exp_val_score or actual_spec.get("fault_score") != exp_fault_score:
                return VerificationResult(False, f"Validation/fault score mismatch for {role}")

            # Verify agent EventDAG provenance chain reaches authoritative source hash
            prov_ref = agent_payload.get("provenance_reference")
            if prov_ref not in events_by_hash:
                return VerificationResult(False, f"Missing agent provenance event for {role}")
            agent_event = events_by_hash[prov_ref]
            if agent_event.get("event_type") != "specialist_assessment":
                return VerificationResult(False, f"Invalid agent provenance event type for {role}")
            trace = ledger.trace(prov_ref)
            if not any(
                e.get("event_type") == "input_received"
                and e.get("payload", {}).get("binding") == authoritative_binding
                for e in trace
            ):
                return VerificationResult(False, f"EventDAG trace does not bind authoritative source for {role}")
            if not any(
                e.get("event_type") == "agent_output"
                and e.get("agent_id") == f"{role}_agent"
                and e.get("payload") == agent_payload.get("raw_output")
                for e in trace
            ):
                return VerificationResult(False, f"EventDAG trace missing matching agent_output for {role}")
            agent_prov_refs.append(prov_ref)

        # 6: Independently evaluate composition truth table under precedence rules
        exp_decision, exp_gate = _expected_composition_decision(short_obligations, precedence)
        if subject_payload.get("decision") != exp_decision:
            return VerificationResult(
                False,
                f"Composed decision mismatch: got {subject_payload.get('decision')}, expected {exp_decision}",
            )
        if subject_payload.get("gate") != exp_gate:
            return VerificationResult(
                False,
                f"Composition gate mismatch: got {subject_payload.get('gate')}, expected {exp_gate}",
            )

        advisory = subject_payload.get("advisory_consensus")
        if isinstance(advisory, dict) and advisory.get("overrode_decision") is not False:
            return VerificationResult(False, "Advisory consensus claimed decision override")

        # Verify composition event in EventDAG matches subject payload and agent provenance parents
        comp_prov_ref = subject_payload.get("provenance_reference")
        if comp_prov_ref not in events_by_hash:
            return VerificationResult(False, "Composition provenance reference not found in EventDAG")
        comp_event = events_by_hash[comp_prov_ref]
        if comp_event.get("event_type") != "specialist_composition":
            return VerificationResult(False, f"Unexpected composition event type: {comp_event.get('event_type')}")
        if comp_event.get("parents") != sorted(set(agent_prov_refs)):
            return VerificationResult(False, "Composition event parent provenance mismatch")
        expected_event_payload = {k: v for k, v in subject_payload.items() if k != "provenance_reference"}
        if canonical_json(comp_event.get("payload")) != canonical_json(expected_event_payload):
            return VerificationResult(False, "Composition event payload mismatch")

        return VerificationResult(True, "All six composition bindings verified independently")
    except (KeyError, TypeError, ValueError) as err:
        return VerificationResult(False, f"Malformed composition round: {err}")
