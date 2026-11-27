"""Bounded evaluation engine comparing specialist composition to legacy baseline and rules-only.

Evaluates:
- Method E: Preserved legacy full prototype (confidence weighting + validation + mandatory gate)
- Method COMP: Specialist composition prototype under substantive exclusion precedence
- Method COMP_STRICT: Specialist composition prototype under strict audit precedence
- Method F: Deterministic rules-only reference (pure oracle)

All annotations are marked SYNTHETIC_PROVISIONAL (no human approval claimed).
Historical experiment records and repaired baseline bundles remain read-only and preserved.
"""
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import json
from math import isfinite
from pathlib import Path
import random

from ..crypto import canonical_json, sha256
from ..oracle.scoring import (
    calculate_metrics,
    calculate_paired_disagreements,
    evaluate_rule_oracle,
    extended_scenarios,
    historical_cases,
    mixed_scenarios,
)
from .composition import (
    COMPOSITION_CONTRACT_VERSION,
    PRECEDENCE_STRICT_AUDIT,
    PRECEDENCE_SUBSTANTIVE_EXCLUSION,
)
from .domain import FAULT_MODES, ROLES, InsuranceClaim, rule_decision
from .workflow import InsuranceAdjudication, SpecialistCompositionAdjudication

EVALUATION_VERSION = "composition-eval-v1"
ANNOTATION_STATUS = "SYNTHETIC_PROVISIONAL"
HUMAN_REVIEWED = False

COMPOSITION_METHOD_REGISTRY = {
    "E": {
        "name": "Legacy Full Prototype",
        "description": "Preserved legacy prototype: confidence weighting + fault detection + mandatory gate (whole-claim voting).",
        "contract": "mandatory-gate-v1",
        "mode": "legacy",
        "has_specialist_composition": False,
        "role_fusion_override": True,
    },
    "COMP": {
        "name": "Specialist-Role Composition (Substantive Exclusion)",
        "description": "New specialist prototype: typed mandatory obligations composed via composition-v1 truth table with substantive exclusion precedence.",
        "contract": COMPOSITION_CONTRACT_VERSION,
        "mode": "composition",
        "precedence": PRECEDENCE_SUBSTANTIVE_EXCLUSION,
        "has_specialist_composition": True,
        "role_fusion_override": False,
    },
    "COMP_STRICT": {
        "name": "Specialist-Role Composition (Strict Audit)",
        "description": "Specialist prototype with strict audit precedence: system uncertainty halts decision into REVIEW.",
        "contract": COMPOSITION_CONTRACT_VERSION,
        "mode": "composition",
        "precedence": PRECEDENCE_STRICT_AUDIT,
        "has_specialist_composition": True,
        "role_fusion_override": False,
    },
    "F": {
        "name": "Deterministic Rules Only",
        "description": "Direct evaluation of domain rules on ground-truth claim facts (pure oracle baseline).",
        "contract": "deterministic-rules-v1",
        "mode": "oracle",
        "has_specialist_composition": False,
        "role_fusion_override": False,
    },
}


def run_composition_study(
    *,
    cases_set: str = "extended",
    rates: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4),
    seeds: tuple[int, ...] = (42, 101, 2024),
) -> dict:
    """Execute bounded offline comparison between Legacy Arm E, Specialist Composition, and Rules-Only."""
    if cases_set == "extended":
        suite = extended_scenarios()
    elif cases_set == "historical":
        suite = historical_cases()
    elif cases_set == "mixed":
        suite = mixed_scenarios()
    else:
        raise ValueError(f"Unknown cases set: {cases_set}")

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

                # 1. Evaluate Legacy Arm E (Full Prototype)
                legacy_wf = InsuranceAdjudication(claim, faults=faults, mode="legacy")
                dec_e = legacy_wf.result()["decision"]

                # 2. Evaluate Specialist Composition (Substantive Exclusion)
                comp_wf = SpecialistCompositionAdjudication(
                    claim, faults=faults, precedence=PRECEDENCE_SUBSTANTIVE_EXCLUSION
                )
                comp_res = comp_wf.result()
                dec_comp = comp_res["decision"]
                gate_comp = comp_res["gate"]
                adv_comp = comp_res["advisory_consensus"]["diagnostic_candidate"]

                # 3. Evaluate Specialist Composition (Strict Audit)
                strict_wf = SpecialistCompositionAdjudication(
                    claim, faults=faults, precedence=PRECEDENCE_STRICT_AUDIT
                )
                dec_strict = strict_wf.result()["decision"]
                gate_strict = strict_wf.result()["gate"]

                # 4. Evaluate Method F: Deterministic Rules Only (Pure Oracle)
                dec_f = evaluate_rule_oracle(claim)

                decisions = {
                    "E": dec_e,
                    "COMP": dec_comp,
                    "COMP_STRICT": dec_strict,
                    "F": dec_f,
                }

                record = {
                    "seed": seed,
                    "requested_rate": rate,
                    "case_idx": case_idx,
                    "case_id": case_id,
                    "label": label,
                    "faults": faults,
                    "affected_agents": len(faults),
                    "decisions": decisions,
                    "comp_gate": gate_comp,
                    "comp_advisory_candidate": adv_comp,
                    "strict_gate": gate_strict,
                }
                seed_trials.append(record)

        seed_summaries[seed] = seed_trials
        trials_data.extend(seed_trials)

    # Aggregations across seeds and rates
    methods = list(COMPOSITION_METHOD_REGISTRY.keys())
    overall_by_method = {}
    for m in methods:
        preds = [t["decisions"][m] for t in trials_data]
        lbls = [t["label"] for t in trials_data]
        overall_by_method[m] = calculate_metrics(preds, lbls)

    # Paired comparisons
    paired_comparisons = {
        "E_vs_COMP": calculate_paired_disagreements(
            [t["decisions"]["E"] for t in trials_data],
            [t["decisions"]["COMP"] for t in trials_data],
        ),
        "COMP_vs_F": calculate_paired_disagreements(
            [t["decisions"]["COMP"] for t in trials_data],
            [t["decisions"]["F"] for t in trials_data],
        ),
        "E_vs_F": calculate_paired_disagreements(
            [t["decisions"]["E"] for t in trials_data],
            [t["decisions"]["F"] for t in trials_data],
        ),
        "COMP_vs_COMP_STRICT": calculate_paired_disagreements(
            [t["decisions"]["COMP"] for t in trials_data],
            [t["decisions"]["COMP_STRICT"] for t in trials_data],
        ),
    }

    # Per-scenario breakdown comparing Arm E, COMP, and F
    scenario_breakdown = {}
    for s in suite:
        cid = s["case_id"]
        lbl = s["label"]
        case_records = [t for t in trials_data if t["case_id"] == cid]
        n_c = len(case_records)

        scenario_breakdown[cid] = {
            "label": lbl,
            "presentations": n_c,
            "E": {
                "correct": sum(r["decisions"]["E"] == lbl for r in case_records),
                "reviews": sum(r["decisions"]["E"] == "REVIEW" for r in case_records),
                "additional_deferrals": sum(r["decisions"]["E"] == "REVIEW" and lbl != "REVIEW" for r in case_records),
            },
            "COMP": {
                "correct": sum(r["decisions"]["COMP"] == lbl for r in case_records),
                "reviews": sum(r["decisions"]["COMP"] == "REVIEW" for r in case_records),
                "additional_deferrals": sum(r["decisions"]["COMP"] == "REVIEW" and lbl != "REVIEW" for r in case_records),
            },
            "F": {
                "correct": sum(r["decisions"]["F"] == lbl for r in case_records),
                "reviews": sum(r["decisions"]["F"] == "REVIEW" for r in case_records),
                "additional_deferrals": 0,
            },
        }

    # Clean rate (rate == 0.0) summary
    clean_records = [t for t in trials_data if t["requested_rate"] == 0.0]
    clean_by_method = {}
    for m in methods:
        preds = [t["decisions"][m] for t in clean_records]
        lbls = [t["label"] for t in clean_records]
        clean_by_method[m] = calculate_metrics(preds, lbls)

    # Investigation of the 42 legacy additional deferrals:
    rejection_records = [t for t in trials_data if t["label"] == "REJECT"]
    rejection_analysis = {
        "total_rejection_presentations": len(rejection_records),
        "legacy_E_reviews": sum(r["decisions"]["E"] == "REVIEW" for r in rejection_records),
        "legacy_E_correct_rejections": sum(r["decisions"]["E"] == "REJECT" for r in rejection_records),
        "composition_reviews": sum(r["decisions"]["COMP"] == "REVIEW" for r in rejection_records),
        "composition_correct_rejections": sum(r["decisions"]["COMP"] == "REJECT" for r in rejection_records),
        "rules_oracle_correct_rejections": sum(r["decisions"]["F"] == "REJECT" for r in rejection_records),
    }

    return {
        "evaluation_version": EVALUATION_VERSION,
        "contract_version": COMPOSITION_CONTRACT_VERSION,
        "annotation_status": ANNOTATION_STATUS,
        "human_reviewed": HUMAN_REVIEWED,
        "case_suite": cases_set,
        "suite_size": num_cases,
        "rates": list(rates),
        "seeds": list(seeds),
        "total_trials": len(trials_data),
        "method_registry": COMPOSITION_METHOD_REGISTRY,
        "overall_by_method": overall_by_method,
        "clean_by_method": clean_by_method,
        "rejection_analysis": rejection_analysis,
        "paired_comparisons": paired_comparisons,
        "scenario_breakdown": scenario_breakdown,
        "trials": trials_data,
    }
