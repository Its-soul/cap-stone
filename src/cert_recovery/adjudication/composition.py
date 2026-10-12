"""Bounded, versioned specialist-role composition prototype.

Contracts:
- composition-v1: Composes validated specialist assessments via explicit truth table.
- Substantive exclusion precedence: Confirmed policy exclusion is authoritative over
  procedural review triggers and non-policy component unavailability.
- Strict audit precedence: System uncertainty halts decision into REVIEW.
- Advisory confidence: Confidence weighting is advisory diagnostic only; zero authority over obligations.
"""
from dataclasses import asdict, dataclass
from math import fsum, isfinite
from typing import Any

from ..crypto import canonical_json
from ..models import VerificationResult
from .consensus import DECISIONS, ConfidenceWeightedConsensus, Vote, assess
from .domain import ROLES, RULES, rule_decision

COMPOSITION_CONTRACT_VERSION = "composition-v1"
ANNOTATION_STATUS = "SYNTHETIC_PROVISIONAL"

ROLE_CONTRACTS = {
    "policy": "coverage-v1",
    "evidence": "documents-v1",
    "risk": "fraud-screen-v1",
    "compliance": "identity-consent-v1",
}

OBLIGATION_NAMES = {
    "policy": "policy_eligibility",
    "evidence": "evidence_integrity",
    "risk": "screening_integrity",
    "compliance": "regulatory_controls",
}

PERMITTED_CONSEQUENCES = {
    "policy": ("APPROVE", "REJECT"),
    "evidence": ("APPROVE", "REVIEW"),
    "risk": ("APPROVE", "REVIEW"),
    "compliance": ("APPROVE", "REVIEW"),
}

IMPOSSIBLE_CONSEQUENCES = {
    "policy": ("REVIEW",),
    "evidence": ("REJECT",),
    "risk": ("REJECT",),
    "compliance": ("REJECT",),
}

PRECEDENCE_SUBSTANTIVE_EXCLUSION = "substantive_exclusion"
PRECEDENCE_STRICT_AUDIT = "strict_audit"
SUPPORTED_PRECEDENCE = (PRECEDENCE_SUBSTANTIVE_EXCLUSION, PRECEDENCE_STRICT_AUDIT)


@dataclass(frozen=True)
class SpecialistAssessment:
    role: str
    contract_version: str
    obligation_name: str
    obligation_result: str       # "SATISFIED", "FAILED", "UNCERTAIN"
    rule_consequence: str        # "APPROVE", "REJECT", "REVIEW"
    evidence_references: dict
    input_binding: dict
    validation_status: str       # "VALID", "INVALID", "FAILED"
    validation_score: float
    fault_score: float
    confidence: float
    confidence_origin: str
    backend_id: str
    reasoning_summary: str
    reasons: tuple[str, ...]

    def __post_init__(self):
        if self.role not in ROLES:
            raise ValueError(f"Unknown role: {self.role}")
        if self.obligation_result not in ("SATISFIED", "FAILED", "UNCERTAIN"):
            raise ValueError(f"Invalid obligation result: {self.obligation_result}")
        if self.rule_consequence not in DECISIONS:
            raise ValueError(f"Invalid rule consequence: {self.rule_consequence}")
        if self.validation_status not in ("VALID", "INVALID", "FAILED"):
            raise ValueError(f"Invalid validation status: {self.validation_status}")


@dataclass(frozen=True)
class CompositionResult:
    decision: str
    gate: str
    precedence: str
    contract_version: str
    specialist_assessments: dict[str, dict]
    advisory_consensus: dict
    procedural_exceptions: tuple[str, ...]
    rule_references: tuple[str, ...]


def assess_specialist(
    output: Any,
    role: str,
    facts: dict,
    binding: dict,
    backend_id: str = "synthetic-rule-agent-v1",
) -> SpecialistAssessment:
    """Assess a specialist role output against its specific obligation contract.

    Distinguishes domain findings (e.g. valid policy exclusion) from execution
    failures, schema violations, stale bindings, or impossible role consequences.
    Unvalidated/failed outputs yield UNCERTAIN / REVIEW and are never treated as satisfied.
    """
    if role not in ROLES:
        raise ValueError(f"Unknown specialist role: {role}")

    contract_version = ROLE_CONTRACTS[role]
    obligation_name = OBLIGATION_NAMES[role]

    # Handle non-dict or explicit execution error
    if not isinstance(output, dict) or "execution_error" in output:
        error_name = output.get("execution_error", "unstructured_output") if isinstance(output, dict) else "non_dict_output"
        return SpecialistAssessment(
            role=role,
            contract_version=contract_version,
            obligation_name=obligation_name,
            obligation_result="UNCERTAIN",
            rule_consequence="REVIEW",
            evidence_references={},
            input_binding=binding,
            validation_status="FAILED",
            validation_score=0.0,
            fault_score=1.0,
            confidence=0.0,
            confidence_origin="execution_failure",
            backend_id=str(output.get("backend", backend_id) if isinstance(output, dict) else backend_id),
            reasoning_summary=f"Specialist output failed execution: {error_name}",
            reasons=("malformed_or_failed_output", f"error:{error_name}"),
        )

    # Basic schema checks
    try:
        conf = output["confidence"]
        if type(conf) not in (int, float) or not isfinite(conf) or not (0.0 <= conf <= 1.0):
            raise ValueError("Invalid confidence")
        confidence = float(conf)
    except (KeyError, TypeError, ValueError):
        return SpecialistAssessment(
            role=role,
            contract_version=contract_version,
            obligation_name=obligation_name,
            obligation_result="UNCERTAIN",
            rule_consequence="REVIEW",
            evidence_references={},
            input_binding=binding,
            validation_status="FAILED",
            validation_score=0.0,
            fault_score=1.0,
            confidence=0.0,
            confidence_origin="schema_error",
            backend_id=str(output.get("backend", backend_id)),
            reasoning_summary="Specialist output failed confidence schema parsing.",
            reasons=("malformed_or_failed_output", "invalid_confidence"),
        )

    raw_decision = output.get("decision")
    if (
        raw_decision not in DECISIONS
        or output.get("agent_id") != role + "_agent"
        or output.get("role") != role
        or not isinstance(output.get("reasoning_summary"), str)
        or not output.get("reasoning_summary")
        or not isinstance(output.get("evidence"), dict)
        or not isinstance(output.get("rule_references"), list)
    ):
        return SpecialistAssessment(
            role=role,
            contract_version=contract_version,
            obligation_name=obligation_name,
            obligation_result="UNCERTAIN",
            rule_consequence="REVIEW",
            evidence_references={},
            input_binding=binding,
            validation_status="FAILED",
            validation_score=0.0,
            fault_score=1.0,
            confidence=confidence,
            confidence_origin="schema_error",
            backend_id=str(output.get("backend", backend_id)),
            reasoning_summary="Specialist output failed structural schema parsing.",
            reasons=("malformed_or_failed_output", "invalid_structured_schema"),
        )

    reasons: list[str] = []

    # Check for impossible role consequences
    if raw_decision in IMPOSSIBLE_CONSEQUENCES[role]:
        reasons.append(f"impossible_role_consequence:{raw_decision}_not_permitted_for_{role}")

    # Provenance and binding validation
    if output.get("input_reference") != binding:
        reasons.append("stale_or_unbound_input")
    if output.get("evidence") != facts:
        reasons.append("evidence_mismatch")
    if output.get("rule_references") != [RULES[role][0]]:
        reasons.append("rule_mismatch")

    # Check against deterministic domain rule decision
    rule_expected = rule_decision(role, facts)
    if raw_decision != rule_expected:
        reasons.append("decision_conflicts_with_rule")

    confidence_origin = "authored-uncalibrated"
    if isinstance(output.get("uncertainty"), dict) and "calibration" in output["uncertainty"]:
        confidence_origin = output["uncertainty"]["calibration"]

    # Handle validation outcome
    if reasons:
        if confidence >= 0.95:
            reasons.append("high_confidence_invalid_output")
        return SpecialistAssessment(
            role=role,
            contract_version=contract_version,
            obligation_name=obligation_name,
            obligation_result="UNCERTAIN",
            rule_consequence="REVIEW",
            evidence_references=output.get("evidence", {}),
            input_binding=binding,
            validation_status="INVALID",
            validation_score=0.0,
            fault_score=1.0,
            confidence=confidence,
            confidence_origin=confidence_origin,
            backend_id=str(output.get("backend", backend_id)),
            reasoning_summary=output.get("reasoning_summary", ""),
            reasons=tuple(reasons),
        )

    # Valid outcome: Map to domain obligation result
    if role == "policy":
        if raw_decision == "APPROVE":
            obligation_result = "SATISFIED"
            rule_consequence = "APPROVE"
        else:  # REJECT
            obligation_result = "FAILED"
            rule_consequence = "REJECT"
    else:
        if raw_decision == "APPROVE":
            obligation_result = "SATISFIED"
            rule_consequence = "APPROVE"
        else:  # REVIEW
            obligation_result = "FAILED"
            rule_consequence = "REVIEW"

    return SpecialistAssessment(
        role=role,
        contract_version=contract_version,
        obligation_name=obligation_name,
        obligation_result=obligation_result,
        rule_consequence=rule_consequence,
        evidence_references=output.get("evidence", {}),
        input_binding=binding,
        validation_status="VALID",
        validation_score=1.0,
        fault_score=0.0,
        confidence=confidence,
        confidence_origin=confidence_origin,
        backend_id=str(output.get("backend", backend_id)),
        reasoning_summary=output.get("reasoning_summary", ""),
        reasons=(),
    )


def compute_advisory_consensus(assessments: dict[str, SpecialistAssessment]) -> dict:
    """Compute confidence-weighted consensus for diagnostic comparison only.

    Contains zero authority to override obligations or issue certificates.
    """
    votes: list[Vote] = []
    for role in ROLES:
        a = assessments[role]
        raw_dec = a.rule_consequence if a.validation_status != "FAILED" else "REVIEW"
        conf = a.confidence if a.validation_status != "FAILED" else 0.0
        votes.append(Vote(
            role + "_agent",
            raw_dec,
            conf,
            validation_score=a.validation_score,
            fault_score=a.fault_score,
        ))

    consensus = ConfidenceWeightedConsensus().aggregate(votes)
    return {
        "diagnostic_candidate": consensus["decision"],
        "weights": consensus["weights"],
        "totals": consensus["totals"],
        "total_weight": consensus["total_weight"],
        "abstained": consensus["abstained"],
        "vote_entropy": consensus["vote_entropy"],
        "role_authority": "ADVISORY_DIAGNOSTIC_ONLY",
        "overrode_decision": False,
        "method": "confidence-weighted-advisory-diagnostic-v1",
    }


def compose_specialist_assessments(
    assessments: dict[str, SpecialistAssessment],
    precedence: str = PRECEDENCE_SUBSTANTIVE_EXCLUSION,
) -> CompositionResult:
    """Compose validated specialist findings according to the frozen truth table.

    Precedence options:
    - substantive_exclusion (Default): Verified policy exclusion is authoritative over
      procedural review triggers and non-policy component unavailability.
    - strict_audit: System uncertainty halts decision into REVIEW.
    """
    if precedence not in SUPPORTED_PRECEDENCE:
        raise ValueError(f"Unsupported composition precedence: {precedence}")
    if set(assessments) != set(ROLES):
        raise ValueError(f"Assessments must bind exactly the four roles: {ROLES}")

    s_p = assessments["policy"]
    s_e = assessments["evidence"]
    s_r = assessments["risk"]
    s_c = assessments["compliance"]

    # Collect non-policy procedural exceptions for audit logging
    exceptions: list[str] = []
    for role in ("evidence", "risk", "compliance"):
        a = assessments[role]
        if a.obligation_result != "SATISFIED":
            exceptions.append(f"{role}:{a.obligation_result}({a.rule_consequence})")

    # Advisory consensus calculation
    advisory = compute_advisory_consensus(assessments)

    decision: str
    gate: str

    if precedence == PRECEDENCE_SUBSTANTIVE_EXCLUSION:
        # Rule 1: Policy Uncertainty
        if s_p.obligation_result == "UNCERTAIN":
            decision = "REVIEW"
            gate = "POLICY_UNCERTAIN"
        # Rule 2: Substantive Exclusion (Authoritative over procedural flags / non-policy uncertainty)
        elif s_p.validation_status == "VALID" and s_p.obligation_result == "FAILED":
            decision = "REJECT"
            gate = "SUBSTANTIVE_EXCLUSION"
        # Rule 3: Procedural Deferral
        elif any(assessments[r].obligation_result in ("FAILED", "UNCERTAIN") for r in ("evidence", "risk", "compliance")):
            decision = "REVIEW"
            gate = "PROCEDURAL_DEFERRAL"
        # Rule 4: All Obligations Satisfied
        elif all(assessments[r].obligation_result == "SATISFIED" for r in ROLES):
            decision = "APPROVE"
            gate = "ALL_OBLIGATIONS_SATISFIED"
        else:
            decision = "REVIEW"
            gate = "FALLTHROUGH_REVIEW"

    elif precedence == PRECEDENCE_STRICT_AUDIT:
        # Rule 1: Audit Uncertainty (Halts decision before domain rules)
        if any(assessments[r].obligation_result == "UNCERTAIN" for r in ROLES):
            decision = "REVIEW"
            gate = "AUDIT_UNCERTAINTY"
        # Rule 2: Substantive Exclusion
        elif s_p.obligation_result == "FAILED":
            decision = "REJECT"
            gate = "SUBSTANTIVE_EXCLUSION"
        # Rule 3: Procedural Deferral
        elif any(assessments[r].obligation_result == "FAILED" for r in ("evidence", "risk", "compliance")):
            decision = "REVIEW"
            gate = "PROCEDURAL_DEFERRAL"
        # Rule 4: All Obligations Satisfied
        elif all(assessments[r].obligation_result == "SATISFIED" for r in ROLES):
            decision = "APPROVE"
            gate = "ALL_OBLIGATIONS_SATISFIED"
        else:
            decision = "REVIEW"
            gate = "FALLTHROUGH_REVIEW"

    advisory["agreed_with_composed"] = (advisory["diagnostic_candidate"] == decision)

    return CompositionResult(
        decision=decision,
        gate=gate,
        precedence=precedence,
        contract_version=COMPOSITION_CONTRACT_VERSION,
        specialist_assessments={r: asdict(assessments[r]) for r in ROLES},
        advisory_consensus=advisory,
        procedural_exceptions=tuple(exceptions),
        rule_references=tuple([RULES[r][0] for r in ROLES] + [COMPOSITION_CONTRACT_VERSION]),
    )


def verify_specialist_composition(
    subject_payload: dict,
    agent_artifacts: tuple[Any, ...],
    ledger: Any,
    precedence: str = PRECEDENCE_SUBSTANTIVE_EXCLUSION,
) -> VerificationResult:
    """Verify composition round artifact against registered verifier contract.

    Checks:
    1. Four distinct roles bound in declared order.
    2. Specialist assessments match independent recomputations from bound facts.
    3. Truth table decision matches independently composed outcome.
    4. EventDAG event hashes and provenance reference binding.
    """
    try:
        if [n.ref.artifact_id for n in agent_artifacts] != [r + "_agent" for r in ROLES]:
            return VerificationResult(False, "Round must bind exactly four distinct roles in declared order")

        recomputed_assessments: dict[str, SpecialistAssessment] = {}
        events = {e["event_hash"]: e for e in ledger.export()}

        for role, node in zip(ROLES, agent_artifacts):
            if len(node.supports) != 1 or node.supports[0].artifact_id != "facts:" + role or node.payload["role"] != role:
                return VerificationResult(False, f"Role/source binding mismatch for {role}")
            source = node.system_source if hasattr(node, "system_source") else None

            # Get facts and binding from payload
            facts = node.payload["evidence"]
            binding = node.payload["input_reference"]
            raw_output = node.payload["raw_output"]

            spec_assessment = assess_specialist(raw_output, role, facts, binding)
            recomputed_assessments[role] = spec_assessment

            expected_spec = asdict(spec_assessment)
            actual_spec = subject_payload["specialist_assessments"][role]
            if canonical_json(expected_spec) != canonical_json(actual_spec):
                return VerificationResult(False, f"Assessment mismatch for {role}")

        # Recompute composition
        expected_composition = compose_specialist_assessments(recomputed_assessments, precedence=precedence)
        if subject_payload["decision"] != expected_composition.decision:
            return VerificationResult(False, "Composed decision mismatch")
        if subject_payload["gate"] != expected_composition.gate:
            return VerificationResult(False, "Composition gate mismatch")
        if subject_payload["contract_version"] != COMPOSITION_CONTRACT_VERSION:
            return VerificationResult(False, "Composition contract version mismatch")

        # Verify EventDAG
        if not ledger.verify(ledger.export(), ledger.checkpoint()):
            return VerificationResult(False, "EventDAG integrity verification failed")

        prov_ref = subject_payload.get("provenance_reference")
        if prov_ref not in events:
            return VerificationResult(False, "Composition provenance reference not found in EventDAG")

        comp_event = events[prov_ref]
        if comp_event["event_type"] != "specialist_composition":
            return VerificationResult(False, f"Unexpected event type: {comp_event['event_type']}")

        # Verify parent references link to each agent's provenance reference
        agent_prov_refs = sorted(set([node.payload["provenance_reference"] for node in agent_artifacts]))
        if comp_event["parents"] != agent_prov_refs:
            return VerificationResult(False, "Specialist composition parent provenance binding mismatch")

        return VerificationResult(True, "Recomputed assessments, truth table composition, and EventDAG verified")
    except (KeyError, TypeError, ValueError) as err:
        return VerificationResult(False, f"Malformed composition round: {err}")
