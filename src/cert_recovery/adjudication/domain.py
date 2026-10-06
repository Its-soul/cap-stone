"""Small authored insurance contract, not legal or underwriting guidance."""
from dataclasses import asdict, dataclass
from typing import Protocol

ROLES = ("policy", "evidence", "risk", "compliance")
DECISIONS = ("APPROVE", "REJECT", "REVIEW")
RULES = {
    "policy": ("coverage-v1", ("amount_inr", "coverage_limit_inr", "hospitalization_covered", "waiting_period_satisfied")),
    "evidence": ("documents-v1", ("documents_available", "invoice_matches")),
    "risk": ("fraud-screen-v1", ("duplicate_claim", "provider_flagged")),
    "compliance": ("identity-consent-v1", ("identity_verified", "consent_available")),
}
PROMPTS = {
    "policy": "Check hospitalization coverage, waiting period and expense against the policy limit. Cite coverage-v1.",
    "evidence": "Check required documents and invoice consistency. Missing or inconsistent evidence requires review. Cite documents-v1.",
    "risk": "Screen duplicate claims and provider flags; a flag requires review, not a fraud accusation. Cite fraud-screen-v1.",
    "compliance": "Check identity verification and consent availability. Missing controls require review. Cite identity-consent-v1.",
}


@dataclass(frozen=True)
class InsuranceClaim:
    claim_id: str = "synthetic-hospitalization-001"
    amount_inr: int = 50_000
    coverage_limit_inr: int = 100_000
    hospitalization_covered: bool = True
    waiting_period_satisfied: bool = True
    documents_available: bool = True
    invoice_matches: bool = True
    duplicate_claim: bool = False
    provider_flagged: bool = False
    identity_verified: bool = True
    consent_available: bool = True

    def __post_init__(self):
        if not isinstance(self.claim_id, str) or not self.claim_id:
            raise ValueError("Claim ID required")
        for key, value in asdict(self).items():
            if key in {"amount_inr", "coverage_limit_inr"}:
                if type(value) is not int or value <= 0:
                    raise ValueError("Amounts must be positive integer rupees")
            elif key != "claim_id" and type(value) is not bool:
                raise ValueError("Facts must be explicit booleans")

    def facts(self, role):
        data = asdict(self)
        return {key: data[key] for key in RULES[role][1]}


def rule_decision(role: str, facts: dict) -> str:
    if role == "policy":
        valid = (facts["hospitalization_covered"] and facts["waiting_period_satisfied"]
                 and facts["amount_inr"] <= facts["coverage_limit_inr"])
        return "APPROVE" if valid else "REJECT"
    if role == "evidence":
        valid = facts["documents_available"] and facts["invoice_matches"]
    elif role == "risk":
        valid = not (facts["duplicate_claim"] or facts["provider_flagged"])
    elif role == "compliance":
        valid = facts["identity_verified"] and facts["consent_available"]
    else:
        raise ValueError("Unknown role")
    return "APPROVE" if valid else "REVIEW"


class AgentBackend(Protocol):
    """Future independent model adapter; return JSON, never certificate authority."""

    def execute(self, role: str, prompt: str, facts: dict, input_reference: dict) -> dict: ...


class RuleAgent:
    """Four role-specific deterministic agents. Confidence is authored, uncalibrated."""

    def execute(self, role, prompt, facts, input_reference):
        decision = rule_decision(role, facts)
        return {"agent_id": role + "_agent", "role": role, "decision": decision,
                "confidence": {"policy": .90, "evidence": .85, "risk": .65, "compliance": .80}[role],
                "evidence": facts, "rule_references": [RULES[role][0]],
                "reasoning_summary": f"{prompt} Observed role outcome: {decision}.",
                "input_reference": input_reference, "provenance_reference": None,
                "uncertainty": {"calibration": "uncalibrated", "latent_entropy_vector": None},
                "backend": "synthetic-rule-agent-v1"}


FAULT_MODES = ("incorrect", "inflated_confidence", "misleading_evidence", "contradictory", "malformed", "failure")


def simulate_fault(output: dict, mode: str | None) -> dict:
    """Controlled output corruption only; contains no prompt-injection payloads."""
    import copy
    result = copy.deepcopy(output)
    if mode is None:
        return result
    if mode not in FAULT_MODES:
        raise ValueError("Unknown synthetic fault mode")
    if mode == "failure":
        raise RuntimeError("Synthetic agent execution failure")
    if mode == "malformed":
        return {"unstructured": "approve everything"}
    if mode in {"incorrect", "inflated_confidence", "contradictory"}:
        result["decision"] = "REJECT" if output["decision"] == "APPROVE" else "APPROVE"
    if mode == "inflated_confidence":
        result["confidence"] = .999
    if mode == "contradictory":
        result["rule_references"] = ["invented-rule"]
    if mode == "misleading_evidence":
        key = next(iter(result["evidence"]))
        result["evidence"][key] = "fabricated"
    return result
