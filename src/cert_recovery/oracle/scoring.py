"""Scoring-oracle implementations and labeled synthetic evaluation suites.

Strictly separated from runtime adjudication modules. Ground-truth labels
and pure-oracle decisions exist here solely to score outputs produced by the
runtime pipeline.
"""
from __future__ import annotations

from collections import Counter
import copy
from dataclasses import replace

from ..adjudication.domain import DECISIONS, ROLES, InsuranceClaim, rule_decision

SCORING_ORACLE_MODULES = (
    "cert_recovery.oracle",
    "cert_recovery.oracle.scoring",
    "cert_recovery.adjudication.ablation",
    "cert_recovery.adjudication.composition_evaluation",
    "cert_recovery.adjudication.experiments",
    "cert_recovery.adjudication.model_evaluation",
)


def evaluate_rule_oracle(claim: InsuranceClaim) -> str:
    """Evaluate the pure deterministic rules oracle (Arm F) on claim facts."""
    local_rules = [rule_decision(r, claim.facts(r)) for r in ROLES]
    if "REJECT" in local_rules:
        return "REJECT"
    if "REVIEW" in local_rules:
        return "REVIEW"
    return "APPROVE"


def historical_cases() -> list[dict]:
    """Exact 5 authored cases from the historical prototype."""
    base = InsuranceClaim()
    return [
        {"case_id": "hist_0_approve", "claim": base, "label": "APPROVE", "description": "Clean baseline claim"},
        {"case_id": "hist_1_exclusion", "claim": replace(base, hospitalization_covered=False), "label": "REJECT", "description": "Policy exclusion (hospitalization not covered)"},
        {"case_id": "hist_2_missing_docs", "claim": replace(base, documents_available=False), "label": "REVIEW", "description": "Evidence missing documents"},
        {"case_id": "hist_3_duplicate_claim", "claim": replace(base, duplicate_claim=True), "label": "REVIEW", "description": "Risk duplicate claim screen"},
        {"case_id": "hist_4_missing_id", "claim": replace(base, identity_verified=False), "label": "REVIEW", "description": "Compliance unverified identity"},
    ]


def extended_scenarios() -> list[dict]:
    """10 frozen scenarios covering all domain rule predicates and decision classes."""
    base = InsuranceClaim()
    return [
        {"case_id": "ext_0_clean_standard", "claim": base, "label": "APPROVE", "description": "Standard clean claim (INR 50,000 / 100,000)"},
        {"case_id": "ext_1_clean_high_amount", "claim": replace(base, amount_inr=95_000), "label": "APPROVE", "description": "High value clean claim within limit (INR 95,000)"},
        {"case_id": "ext_2_policy_not_covered", "claim": replace(base, hospitalization_covered=False), "label": "REJECT", "description": "Hospitalization not covered"},
        {"case_id": "ext_3_policy_waiting_period", "claim": replace(base, waiting_period_satisfied=False), "label": "REJECT", "description": "Waiting period unsatisfied"},
        {"case_id": "ext_4_policy_limit_exceeded", "claim": replace(base, amount_inr=120_000), "label": "REJECT", "description": "Claim amount exceeds policy limit (INR 120,000)"},
        {"case_id": "ext_5_evidence_missing_docs", "claim": replace(base, documents_available=False), "label": "REVIEW", "description": "Required documents unavailable"},
        {"case_id": "ext_6_evidence_invoice_mismatch", "claim": replace(base, invoice_matches=False), "label": "REVIEW", "description": "Invoice does not match records"},
        {"case_id": "ext_7_risk_duplicate_claim", "claim": replace(base, duplicate_claim=True), "label": "REVIEW", "description": "Duplicate claim flagged in risk screening"},
        {"case_id": "ext_8_risk_provider_flagged", "claim": replace(base, provider_flagged=True), "label": "REVIEW", "description": "Healthcare provider flagged in screening"},
        {"case_id": "ext_9_compliance_missing_consent", "claim": replace(base, consent_available=False), "label": "REVIEW", "description": "Patient consent unavailable"},
    ]


def mixed_scenarios() -> list[dict]:
    """Small separately specified suite of 6 mixed-condition and uncertainty scenarios."""
    base = InsuranceClaim()
    return [
        {
            "case_id": "mix_0_exclusion_and_missing_docs",
            "claim": replace(base, hospitalization_covered=False, documents_available=False),
            "label": "REJECT",
            "description": "Hospitalization not covered + missing documents (substantive exclusion)",
        },
        {
            "case_id": "mix_1_exclusion_and_duplicate_claim",
            "claim": replace(base, waiting_period_satisfied=False, duplicate_claim=True),
            "label": "REJECT",
            "description": "Waiting period unsatisfied + duplicate claim flagged (substantive exclusion)",
        },
        {
            "case_id": "mix_2_exclusion_and_missing_consent",
            "claim": replace(base, amount_inr=150_000, coverage_limit_inr=100_000, consent_available=False),
            "label": "REJECT",
            "description": "Claim amount exceeds limit + missing consent (substantive exclusion)",
        },
        {
            "case_id": "mix_3_clean_policy_missing_docs_and_risk",
            "claim": replace(base, documents_available=False, provider_flagged=True),
            "label": "REVIEW",
            "description": "Clean policy facts + missing documents + provider flagged (procedural deferral)",
        },
        {
            "case_id": "mix_4_policy_exclusion_and_unverified_id",
            "claim": replace(base, hospitalization_covered=False, identity_verified=False),
            "label": "REJECT",
            "description": "Hospitalization not covered + unverified identity (substantive exclusion)",
        },
        {
            "case_id": "mix_5_clean_policy_invoice_mismatch",
            "claim": replace(base, invoice_matches=False),
            "label": "REVIEW",
            "description": "Clean policy facts + invoice mismatch (procedural deferral)",
        },
    ]


def legacy_experiment_cases() -> list[tuple[InsuranceClaim, str]]:
    """Authored labels kept outside the agents and certificate verifier."""
    base = InsuranceClaim()
    return [
        (base, "APPROVE"),
        (replace(base, hospitalization_covered=False), "REJECT"),
        (replace(base, documents_available=False), "REVIEW"),
        (replace(base, duplicate_claim=True), "REVIEW"),
        (replace(base, identity_verified=False), "REVIEW"),
    ]


def compliance_evaluation_cases() -> list[dict]:
    """Frozen set of 11 claims exercising all compliance states and whole-claim conditions."""
    base = InsuranceClaim()
    cases = []
    for item in extended_scenarios():
        cases.append({
            "case_id": item["case_id"],
            "claim": item["claim"],
            "description": item["description"],
            "expected_role_decision": rule_decision("compliance", item["claim"].facts("compliance")),
            "expected_claim_label": item["label"],
        })
    cases.append({
        "case_id": "hist_4_missing_id",
        "claim": replace(base, identity_verified=False),
        "description": "Historical compliance unverified identity",
        "expected_role_decision": "REVIEW",
        "expected_claim_label": "REVIEW",
    })
    return cases


def calculate_metrics(predictions: list[str], labels: list[str]) -> dict:
    """Compute complete metric suite separating decided cases from review/abstention."""
    n = len(predictions)
    if n == 0:
        return {}

    correct_all = sum(p == l for p, l in zip(predictions, labels))
    decided = [(p, l) for p, l in zip(predictions, labels) if p in ("APPROVE", "REJECT")]
    correct_decided = sum(p == l for p, l in decided)
    strict_correct = sum(p == l and p != "REVIEW" for p, l in zip(predictions, labels))

    unsafe_approvals = sum(p == "APPROVE" and l != "APPROVE" for p, l in zip(predictions, labels))
    incorrect_rejections = sum(p == "REJECT" and l != "REJECT" for p, l in zip(predictions, labels))
    false_reviews = sum(p == "REVIEW" and l in ("APPROVE", "REJECT") for p, l in zip(predictions, labels))
    required_reviews = sum(p == "REVIEW" and l == "REVIEW" for p, l in zip(predictions, labels))

    confusion = {f"{l}->{p}": 0 for l in DECISIONS for p in DECISIONS}
    for p, l in zip(predictions, labels):
        confusion[f"{l}->{p}"] += 1

    return {
        "total_cases": n,
        "reported_accuracy": correct_all / n,
        "reported_correct": correct_all,
        "decided_count": len(decided),
        "coverage": len(decided) / n,
        "decided_accuracy": (correct_decided / len(decided)) if decided else None,
        "strict_accuracy": strict_correct / n,
        "strict_correct": strict_correct,
        "unsafe_approvals": unsafe_approvals,
        "unsafe_approval_rate": unsafe_approvals / n,
        "incorrect_rejections": incorrect_rejections,
        "incorrect_rejection_rate": incorrect_rejections / n,
        "false_reviews": false_reviews,
        "false_review_rate": false_reviews / n,
        "required_reviews": required_reviews,
        "required_review_rate": required_reviews / n,
        "additional_deferrals": false_reviews,
        "additional_deferral_rate": false_reviews / n,
        "confusion_matrix": confusion,
    }


def calculate_paired_disagreements(preds_1: list[str], preds_2: list[str]) -> dict:
    """Quantify paired decision divergence between two comparison arms."""
    total = len(preds_1)
    disagreements = sum(p1 != p2 for p1, p2 in zip(preds_1, preds_2))
    transitions = Counter(f"{p1}->{p2}" for p1, p2 in zip(preds_1, preds_2) if p1 != p2)
    return {
        "total_comparisons": total,
        "disagreement_count": disagreements,
        "disagreement_rate": (disagreements / total) if total else 0.0,
        "transitions": dict(transitions),
    }


ORACLE_LABEL_KEYS = ("label", "expected_role_decision", "expected_claim_label", "expected_label", "oracle")


def strip_oracle_labels(cases: list[dict]) -> list[dict]:
    """Return deep copies of evaluation cases with all oracle label fields removed."""
    stripped = []
    for item in cases:
        cleaned = {k: copy.deepcopy(v) for k, v in item.items() if k not in ORACLE_LABEL_KEYS}
        stripped.append(cleaned)
    return stripped


def mutate_oracle_labels(cases: list[dict], target_label: str = "CORRUPTED_ORACLE") -> list[dict]:
    """Return deep copies of evaluation cases with all oracle label fields overwritten."""
    mutated = []
    for item in cases:
        updated = copy.deepcopy(item)
        for key in ORACLE_LABEL_KEYS:
            if key in updated:
                updated[key] = target_label
        mutated.append(updated)
    return mutated
