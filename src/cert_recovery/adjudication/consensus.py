"""Deterministic weighted voting, not a Byzantine agreement protocol."""
from dataclasses import dataclass
from math import fsum, isfinite, log
from typing import Protocol

from .domain import DECISIONS, RULES, rule_decision


def probability(value):
    if type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Expected finite probability in [0,1]")
    return float(value)


class ConfidenceCalibrator(Protocol):
    """Fit on held-out role-labelled data outside voting; freeze before evaluation."""

    def transform(self, agent_id: str, confidence: float) -> float: ...


@dataclass(frozen=True)
class Assessment:
    validation_status: str
    validation_score: float
    fault_score: float
    trust_status: str
    reasons: tuple[str, ...]


def assess(output, role, facts, binding) -> Assessment:
    """Checks declared evidence/rules, not malicious intent or prose faithfulness."""
    try:
        probability(output["confidence"])
        if (output["decision"] not in DECISIONS or output["agent_id"] != role + "_agent"
                or output["role"] != role or not isinstance(output["reasoning_summary"], str)
                or not output["reasoning_summary"] or not isinstance(output["evidence"], dict)
                or not isinstance(output["rule_references"], list)):
            raise ValueError("Invalid structured output")
        reasons = []
        if output["input_reference"] != binding:
            reasons.append("stale_or_unbound_input")
        if output["evidence"] != facts:
            reasons.append("evidence_mismatch")
        if output["rule_references"] != [RULES[role][0]]:
            reasons.append("rule_mismatch")
        if output["decision"] != rule_decision(role, facts):
            reasons.append("decision_conflicts_with_rule")
        if reasons:
            if output["confidence"] >= .95:
                reasons.append("high_confidence_invalid_output")
            trust = "INCORRECT" if reasons == ["decision_conflicts_with_rule"] else "SUSPICIOUS"
            return Assessment("INVALID", 0., 1., trust, tuple(reasons))
        return Assessment("VALID", 1., 0., "VALIDATED", ())
    except (KeyError, TypeError, ValueError):
        return Assessment("FAILED", 0., 1., "CONFIRMED_FAILURE", ("malformed_or_failed_output",))


@dataclass(frozen=True)
class Vote:
    agent_id: str
    decision: str
    confidence: float
    validation_score: float = 1.
    fault_score: float = 0.
    reliability: float = 1.

    def __post_init__(self):
        if not self.agent_id or self.decision not in DECISIONS:
            raise ValueError("Invalid vote identity or decision")
        for value in (self.confidence, self.validation_score, self.fault_score, self.reliability):
            probability(value)


class ConfidenceWeightedConsensus:
    def __init__(self, calibrator: ConfidenceCalibrator | None = None):
        self.calibrator = calibrator

    def aggregate(self, votes: list[Vote]) -> dict:
        if len({v.agent_id for v in votes}) != len(votes):
            raise ValueError("Duplicate agent vote")
        weights = {}
        for vote in sorted(votes, key=lambda v: v.agent_id):
            confidence = probability(self.calibrator.transform(vote.agent_id, vote.confidence)
                                     if self.calibrator else vote.confidence)
            weights[vote.agent_id] = confidence * vote.reliability * vote.validation_score * (1 - vote.fault_score)
        totals = {d: fsum(weights[v.agent_id] for v in sorted(votes, key=lambda v: v.agent_id)
                          if v.decision == d) for d in DECISIONS}
        total = fsum(totals.values())
        winners = [d for d in DECISIONS if totals[d] == max(totals.values())]
        decision = winners[0] if total and len(winners) == 1 else "REVIEW"
        shares = [weight / total for weight in totals.values() if weight] if total else []
        return {"decision": decision, "weights": weights, "totals": totals,
                "total_weight": total, "abstained": not total or len(winners) != 1,
                "vote_entropy": -fsum(p * log(p) for p in shares),
                "calibration_transform_applied": self.calibrator is not None,
                "calibration_status": "external_transform_unverified" if self.calibrator else "uncalibrated",
                "method": "confidence-weighted-prototype-v1"}


def majority(outputs: list[dict]) -> str:
    """Same raw outputs; malformed decisions abstain. Ties go to REVIEW."""
    counts = {d: sum(isinstance(o, dict) and o.get("decision") == d for o in outputs) for d in DECISIONS}
    winners = [d for d in DECISIONS if counts[d] == max(counts.values())]
    return winners[0] if max(counts.values()) and len(winners) == 1 else "REVIEW"
