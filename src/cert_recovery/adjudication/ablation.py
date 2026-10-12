"""Bounded offline ablation study comparing consensus, gate, and detection arms.

This module enforces a strict frozen evaluation protocol:
- Annotations are marked SYNTHETIC_PROVISIONAL (no human review claimed).
- Historical cases and results are strictly preserved.
- Methods A-F are evaluated on identical agent outputs and fault schedules.
- Information access is documented explicitly for every comparison arm.
"""
from collections import Counter, defaultdict
import copy
from dataclasses import asdict, replace
import json
from math import isfinite, sqrt
from pathlib import Path
import random

from ..crypto import canonical_json, sha256
from ..metrics import ratio
from .consensus import ConfidenceWeightedConsensus, Vote, majority
from .domain import DECISIONS, FAULT_MODES, ROLES, RULES, InsuranceClaim, rule_decision
from .workflow import InsuranceAdjudication

SPEC_VERSION = "ablation-v1"
ANNOTATION_STATUS = "SYNTHETIC_PROVISIONAL"
HUMAN_REVIEWED = False

# Information access disclosure per comparison arm
METHOD_REGISTRY = {
    "A": {
        "name": "Majority only",
        "description": "Unweighted majority of raw agent outputs.",
        "inputs_accessed": ["raw_output.decision"],
        "oracle_access": "NONE",
        "has_gate": False,
        "has_validation": False,
        "has_confidence": False,
    },
    "B": {
        "name": "Majority + Mandatory Gate",
        "description": "Unweighted majority filtered by the deterministic rule gate.",
        "inputs_accessed": ["raw_output.decision", "facts", "rule_decision"],
        "oracle_access": "GATE_ONLY",
        "has_gate": True,
        "has_validation": False,
        "has_confidence": False,
    },
    "C": {
        "name": "Confidence Weighting only",
        "description": "Confidence-weighted consensus of raw outputs without validation.",
        "inputs_accessed": ["raw_output.decision", "raw_output.confidence"],
        "oracle_access": "NONE",
        "has_gate": False,
        "has_validation": False,
        "has_confidence": True,
    },
    "D": {
        "name": "Confidence Weighting + Fault Detection",
        "description": "Confidence-weighted consensus filtered by output validation (no gate).",
        "inputs_accessed": ["raw_output", "facts", "binding", "rule_decision"],
        "oracle_access": "DETECTOR_ONLY",
        "has_gate": False,
        "has_validation": True,
        "has_confidence": True,
    },
    "E": {
        "name": "Full Prototype",
        "description": "Confidence weighting + fault detection + mandatory gate + certificates.",
        "inputs_accessed": ["raw_output", "facts", "binding", "rule_decision", "event_dag"],
        "oracle_access": "DETECTOR_AND_GATE",
        "has_gate": True,
        "has_validation": True,
        "has_confidence": True,
    },
    "F": {
        "name": "Deterministic Rules only",
        "description": "Direct evaluation of domain rules on ground-truth claim facts (pure oracle).",
        "inputs_accessed": ["facts", "rule_decision"],
        "oracle_access": "PURE_ORACLE",
        "has_gate": True,
        "has_validation": False,
        "has_confidence": False,
    },
    "G": {
        "name": "Majority + Validation + Gate",
        "description": "Unweighted majority of validated agent outputs filtered by the deterministic rule gate.",
        "inputs_accessed": ["raw_output.decision", "raw_output", "facts", "binding", "rule_decision"],
        "oracle_access": "DETECTOR_AND_GATE",
        "has_gate": True,
        "has_validation": True,
        "has_confidence": False,
    },
    "H": {
        "name": "Majority + Validation (no gate)",
        "description": "Unweighted majority of validated agent outputs without safety gate.",
        "inputs_accessed": ["raw_output.decision", "raw_output", "facts", "binding", "rule_decision"],
        "oracle_access": "DETECTOR_ONLY",
        "has_gate": False,
        "has_validation": True,
        "has_confidence": False,
    },
    "I": {
        "name": "Weighting + Gate (no validation)",
        "description": "Confidence-weighted consensus of raw outputs filtered by the deterministic rule gate.",
        "inputs_accessed": ["raw_output.decision", "raw_output.confidence", "facts", "rule_decision"],
        "oracle_access": "GATE_ONLY",
        "has_gate": True,
        "has_validation": False,
        "has_confidence": True,
    },
}


def historical_cases():
    """Exact 5 authored cases from the historical prototype."""
    base = InsuranceClaim()
    return [
        {"case_id": "hist_0_approve", "claim": base, "label": "APPROVE", "description": "Clean baseline claim"},
        {"case_id": "hist_1_exclusion", "claim": replace(base, hospitalization_covered=False), "label": "REJECT", "description": "Policy exclusion (hospitalization not covered)"},
        {"case_id": "hist_2_missing_docs", "claim": replace(base, documents_available=False), "label": "REVIEW", "description": "Evidence missing documents"},
        {"case_id": "hist_3_duplicate_claim", "claim": replace(base, duplicate_claim=True), "label": "REVIEW", "description": "Risk duplicate claim screen"},
        {"case_id": "hist_4_missing_id", "claim": replace(base, identity_verified=False), "label": "REVIEW", "description": "Compliance unverified identity"},
    ]


def extended_scenarios():
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


def evaluate_arms(claim: InsuranceClaim, data: dict) -> dict[str, str]:
    """Evaluate methods A through F on the exact same agent outputs and facts."""
    # Method A: Majority only
    raw_outputs = [a["raw_output"] for a in data["agents"]]
    dec_a = majority(raw_outputs)

    # Oracle rule outcome for gate
    local_rules = [rule_decision(r, claim.facts(r)) for r in ROLES]
    oracle_required = "REJECT" if "REJECT" in local_rules else "REVIEW" if "REVIEW" in local_rules else "APPROVE"

    # Method B: Majority + Gate
    dec_b = dec_a if dec_a == oracle_required else "REVIEW"

    # Method C: Confidence Weighting only (unvalidated)
    unvalidated_votes = []
    for a in data["agents"]:
        raw = a.get("raw_output", {})
        if isinstance(raw, dict) and raw.get("decision") in DECISIONS:
            conf = raw.get("confidence", 0.0)
            try:
                conf = float(conf)
                if not (0.0 <= conf <= 1.0 and isfinite(conf)):
                    conf = 0.0
            except (TypeError, ValueError):
                conf = 0.0
            unvalidated_votes.append(Vote(a["agent_id"], raw["decision"], conf, validation_score=1.0, fault_score=0.0))
        else:
            unvalidated_votes.append(Vote(a["agent_id"], "REVIEW", 0.0, validation_score=1.0, fault_score=0.0))
    dec_c = ConfidenceWeightedConsensus().aggregate(unvalidated_votes)["decision"]

    # Method D: Confidence Weighting + Fault Detection (no gate)
    dec_d = data["round"]["consensus"]["decision"]

    # Method E: Full Prototype (Weighted + Detection + Gate)
    dec_e = data["final"]["decision"]

    # Method F: Deterministic Rules only
    dec_f = oracle_required

    # Method H: Majority + Validation (no gate)
    validated_raw_outputs = [
        a["raw_output"] for a in data["agents"]
        if a.get("assessment", {}).get("validation_status") == "VALID"
    ]
    dec_h = majority(validated_raw_outputs)

    # Method G: Majority + Validation + Gate
    dec_g = dec_h if dec_h == oracle_required else "REVIEW"

    # Method I: Weighting + Gate (no validation)
    dec_i = dec_c if dec_c == oracle_required else "REVIEW"

    return {
        "A": dec_a,
        "B": dec_b,
        "C": dec_c,
        "D": dec_d,
        "E": dec_e,
        "F": dec_f,
        "G": dec_g,
        "H": dec_h,
        "I": dec_i,
    }


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


def run_ablation_study(
    *,
    cases_set: str = "extended",
    rates: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4),
    seeds: tuple[int, ...] = (42, 101, 2024),
) -> dict:
    """Execute bounded offline ablation study across all 6 methods and requested seeds."""
    suite = extended_scenarios() if cases_set == "extended" else historical_cases()
    num_cases = len(suite)

    trials_data = []
    seed_summaries = {}

    for seed in seeds:
        rng = random.Random(seed)
        seed_trials = []

        for rate in rates:
            for case_idx, item in enumerate(suite):
                claim = item["claim"]
                label = item["label"]
                case_id = item["case_id"]

                # Sample independent Bernoulli faults per role
                faults = {r + "_agent": rng.choice(FAULT_MODES) for r in ROLES if rng.random() < rate}

                workflow = InsuranceAdjudication(claim, faults=faults)
                data = workflow.export()

                decisions = evaluate_arms(claim, data)

                flagged = {a["agent_id"] for a in data["agents"] if a["assessment"]["trust_status"] != "VALIDATED"}
                tp = len(flagged & faults.keys())
                fp = len(flagged - faults.keys())

                record = {
                    "seed": seed,
                    "requested_rate": rate,
                    "case_idx": case_idx,
                    "case_id": case_id,
                    "label": label,
                    "faults": faults,
                    "affected_agents": len(faults),
                    "flagged_agents": sorted(flagged),
                    "true_positive": tp,
                    "false_positive": fp,
                    "decisions": decisions,
                }
                seed_trials.append(record)

        seed_summaries[seed] = seed_trials
        trials_data.extend(seed_trials)

    # Aggregations across seeds and rates
    methods = list(METHOD_REGISTRY.keys())
    overall_by_method = {}
    paired_comparisons = {}

    for m in methods:
        preds = [t["decisions"][m] for t in trials_data]
        lbls = [t["label"] for t in trials_data]
        overall_by_method[m] = calculate_metrics(preds, lbls)

    # Paired disagreements between key arms (including full factorial single-factor pairs)
    paired_keys = [
        # Gate effect (Gate ON vs Gate OFF)
        ("A", "B"), ("H", "G"), ("C", "I"), ("D", "E"),
        # Validation effect (Val ON vs Val OFF)
        ("A", "H"), ("B", "G"), ("C", "D"), ("I", "E"),
        # Weighting effect (Weighted vs Majority)
        ("A", "C"), ("B", "I"), ("H", "D"), ("G", "E"),
        # Reference comparison (Rules Oracle vs Full Prototype)
        ("E", "F"),
    ]
    for m1, m2 in paired_keys:
        p1 = [t["decisions"][m1] for t in trials_data]
        p2 = [t["decisions"][m2] for t in trials_data]
        paired_comparisons[f"{m1}_vs_{m2}"] = calculate_paired_disagreements(p1, p2)

    # Single-factor matched differences isolating one factor at a time
    single_factor_effects = {
        "gate_effect": {
            "description": "Gate ON vs Gate OFF across 4 matched consensus/validation settings",
            "pairs": {
                "majority_unvalidated": {"off": "A", "on": "B", "delta_accuracy": overall_by_method["B"]["reported_accuracy"] - overall_by_method["A"]["reported_accuracy"], "delta_unsafe": overall_by_method["B"]["unsafe_approvals"] - overall_by_method["A"]["unsafe_approvals"]},
                "majority_validated": {"off": "H", "on": "G", "delta_accuracy": overall_by_method["G"]["reported_accuracy"] - overall_by_method["H"]["reported_accuracy"], "delta_unsafe": overall_by_method["G"]["unsafe_approvals"] - overall_by_method["H"]["unsafe_approvals"]},
                "weighted_unvalidated": {"off": "C", "on": "I", "delta_accuracy": overall_by_method["I"]["reported_accuracy"] - overall_by_method["C"]["reported_accuracy"], "delta_unsafe": overall_by_method["I"]["unsafe_approvals"] - overall_by_method["C"]["unsafe_approvals"]},
                "weighted_validated": {"off": "D", "on": "E", "delta_accuracy": overall_by_method["E"]["reported_accuracy"] - overall_by_method["D"]["reported_accuracy"], "delta_unsafe": overall_by_method["E"]["unsafe_approvals"] - overall_by_method["D"]["unsafe_approvals"]},
            },
        },
        "validation_effect": {
            "description": "Validation ON vs Validation OFF across 4 matched consensus/gate settings",
            "pairs": {
                "majority_ungated": {"off": "A", "on": "H", "delta_accuracy": overall_by_method["H"]["reported_accuracy"] - overall_by_method["A"]["reported_accuracy"], "delta_unsafe": overall_by_method["H"]["unsafe_approvals"] - overall_by_method["A"]["unsafe_approvals"]},
                "majority_gated": {"off": "B", "on": "G", "delta_accuracy": overall_by_method["G"]["reported_accuracy"] - overall_by_method["B"]["reported_accuracy"], "delta_unsafe": overall_by_method["G"]["unsafe_approvals"] - overall_by_method["B"]["unsafe_approvals"]},
                "weighted_ungated": {"off": "C", "on": "D", "delta_accuracy": overall_by_method["D"]["reported_accuracy"] - overall_by_method["C"]["reported_accuracy"], "delta_unsafe": overall_by_method["D"]["unsafe_approvals"] - overall_by_method["C"]["unsafe_approvals"]},
                "weighted_gated": {"off": "I", "on": "E", "delta_accuracy": overall_by_method["E"]["reported_accuracy"] - overall_by_method["I"]["reported_accuracy"], "delta_unsafe": overall_by_method["E"]["unsafe_approvals"] - overall_by_method["I"]["unsafe_approvals"]},
            },
        },
        "weighting_effect": {
            "description": "Weighting vs Majority across 4 matched validation/gate settings",
            "pairs": {
                "unvalidated_ungated": {"majority": "A", "weighted": "C", "delta_accuracy": overall_by_method["C"]["reported_accuracy"] - overall_by_method["A"]["reported_accuracy"], "delta_unsafe": overall_by_method["C"]["unsafe_approvals"] - overall_by_method["A"]["unsafe_approvals"]},
                "unvalidated_gated": {"majority": "B", "weighted": "I", "delta_accuracy": overall_by_method["I"]["reported_accuracy"] - overall_by_method["B"]["reported_accuracy"], "delta_unsafe": overall_by_method["I"]["unsafe_approvals"] - overall_by_method["B"]["unsafe_approvals"]},
                "validated_ungated": {"majority": "H", "weighted": "D", "delta_accuracy": overall_by_method["D"]["reported_accuracy"] - overall_by_method["H"]["reported_accuracy"], "delta_unsafe": overall_by_method["D"]["unsafe_approvals"] - overall_by_method["H"]["unsafe_approvals"]},
                "validated_gated": {"majority": "G", "weighted": "E", "delta_accuracy": overall_by_method["E"]["reported_accuracy"] - overall_by_method["G"]["reported_accuracy"], "delta_unsafe": overall_by_method["E"]["unsafe_approvals"] - overall_by_method["G"]["unsafe_approvals"]},
            },
        },
    }

    # Rate-level aggregations
    rate_summaries = []
    for rate in rates:
        rate_trials = [t for t in trials_data if t["requested_rate"] == rate]
        total_agent_slots = len(rate_trials) * 4
        affected_agents = sum(t["affected_agents"] for t in rate_trials)
        tp_sum = sum(t["true_positive"] for t in rate_trials)
        fp_sum = sum(t["false_positive"] for t in rate_trials)

        rate_entry = {
            "requested_rate": rate,
            "realized_rate": (affected_agents / total_agent_slots) if total_agent_slots else 0.0,
            "total_trials": len(rate_trials),
            "total_agent_slots": total_agent_slots,
            "affected_agents": affected_agents,
            "detection_recall": (tp_sum / affected_agents) if affected_agents else None,
            "false_suspicion_rate": (fp_sum / (total_agent_slots - affected_agents)) if (total_agent_slots - affected_agents) else 0.0,
            "methods": {},
        }
        for m in methods:
            p_m = [t["decisions"][m] for t in rate_trials]
            l_m = [t["label"] for t in rate_trials]
            rate_entry["methods"][m] = calculate_metrics(p_m, l_m)
        rate_summaries.append(rate_entry)

    # Scenario-level aggregations
    scenario_summaries = {}
    for item in suite:
        cid = item["case_id"]
        c_trials = [t for t in trials_data if t["case_id"] == cid]
        s_entry = {
            "case_id": cid,
            "label": item["label"],
            "description": item["description"],
            "total_trials": len(c_trials),
            "methods": {},
        }
        for m in methods:
            p_m = [t["decisions"][m] for t in c_trials]
            l_m = [t["label"] for t in c_trials]
            s_entry["methods"][m] = calculate_metrics(p_m, l_m)
        scenario_summaries[cid] = s_entry

    # Seed-level variation
    seed_stats = {}
    for seed in seeds:
        s_trials = [t for t in trials_data if t["seed"] == seed]
        seed_stats[seed] = {m: calculate_metrics([t["decisions"][m] for t in s_trials], [t["label"] for t in s_trials])
                            for m in methods}

    return {
        "spec_version": SPEC_VERSION,
        "annotation_status": ANNOTATION_STATUS,
        "human_reviewed": HUMAN_REVIEWED,
        "case_suite": cases_set,
        "suite_size": num_cases,
        "rates": list(rates),
        "seeds": list(seeds),
        "total_trials": len(trials_data),
        "method_registry": METHOD_REGISTRY,
        "overall_by_method": overall_by_method,
        "single_factor_effects": single_factor_effects,
        "rate_summaries": rate_summaries,
        "scenario_summaries": scenario_summaries,
        "paired_comparisons": paired_comparisons,
        "seed_variation": seed_stats,
        "trials": trials_data,
    }


def save_ablation_results(results: dict, output_path: str | Path) -> Path:
    """Save ablation results exclusively to prevent overwriting existing files."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, allow_nan=False)
    return path
