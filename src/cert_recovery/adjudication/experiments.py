"""Seeded, offline output-corruption experiments; not prompt-injection evidence."""
import random
from dataclasses import replace

from ..metrics import ratio
from .domain import FAULT_MODES, ROLES, InsuranceClaim
from .workflow import InsuranceAdjudication


def cases():
    """Authored labels kept outside the agents and certificate verifier."""
    base = InsuranceClaim()
    return [(base, "APPROVE"),
            (replace(base, hospitalization_covered=False), "REJECT"),
            (replace(base, documents_available=False), "REVIEW"),
            (replace(base, duplicate_claim=True), "REVIEW"),
            (replace(base, identity_verified=False), "REVIEW")]


def run_experiments(*, trials=20, seed=42, rates=(0., .1, .2, .3, .4)):
    if type(trials) is not int or trials < 1:
        raise ValueError("Positive trial count required")
    if any(type(r) not in (int, float) or not 0 <= r <= 1 for r in rates):
        raise ValueError("Rates must be in [0,1]")
    rng = random.Random(seed)
    rows, summaries = [], []
    for rate in rates:
        batch = []
        for index in range(trials):
            claim, label = cases()[index % len(cases())]
            # Bernoulli per agent: four-agent trials cannot individually realize 10% or 40%.
            faults = {r + "_agent": rng.choice(FAULT_MODES) for r in ROLES if rng.random() < rate}
            workflow = InsuranceAdjudication(claim, faults=faults)
            data = workflow.export()
            flagged = {a["agent_id"] for a in data["agents"] if a["assessment"]["trust_status"] != "VALIDATED"}
            before = dict(workflow.executions)
            outcomes = []
            for agent_id in faults:
                outcome = workflow.recover_agent(agent_id)
                outcomes.append(outcome.status == "COMPLETE" and
                    workflow.system.latest(agent_id).payload["assessment"]["trust_status"] == "VALIDATED" and
                    workflow.system.is_valid(workflow.system.latest("final").ref))
            recovery_executions = sum(workflow.executions[r] - before[r] for r in ROLES)
            row = {"requested_rate": rate, "trial": index, "case_index": index % len(cases()), "label": label,
                   "faults": faults, "affected_agents": len(faults), "flagged": sorted(flagged),
                   "true_positive": len(flagged & faults.keys()), "false_positive": len(flagged - faults.keys()),
                   "baseline": data["round"]["baseline"], "weighted": data["round"]["consensus"]["decision"],
                   "proposed": data["final"]["decision"], "gate": data["final"]["gate"],
                   "recovery_success": all(outcomes) if faults else None,
                   "recovery_executions": recovery_executions,
                   "post_recovery_decision": workflow.result()["decision"]}
            batch.append(row)
        affected = sum(r["affected_agents"] for r in batch)
        recoveries = [r for r in batch if r["recovery_success"] is not None]
        summaries.append({"requested_rate": rate, "realized_rate": affected / (4 * trials), "trials": trials,
            "affected_agents": affected,
            "baseline_accuracy": sum(r["baseline"] == r["label"] for r in batch) / trials,
            "weighted_accuracy": sum(r["weighted"] == r["label"] for r in batch) / trials,
            "proposed_accuracy": sum(r["proposed"] == r["label"] for r in batch) / trials,
            "detection_recall": ratio(sum(r["true_positive"] for r in batch), affected),
            "false_suspicion_rate": ratio(sum(r["false_positive"] for r in batch), 4 * trials - affected),
            "recovery_success_rate": ratio(sum(r["recovery_success"] for r in recoveries), len(recoveries)),
            "recovery_trials": len(recoveries)})
        rows.extend(batch)
    return {"synthetic": True, "seed": seed, "sampling": "independent Bernoulli per role per trial",
            "limitations": "Authored rules and detectable output corruption; no LLM inference, prompt injection, calibration or BFT guarantee. Recovery uses known injected identities and a repaired backend. Role-local majority is an intentionally weak baseline; report weighted-only ablation separately.",
            "summaries": summaries, "trials": rows}


def comparison():
    scenarios = {"all_correct": {}, "one_incorrect": {"risk_agent": "incorrect"},
                 "high_confidence_fault": {"risk_agent": "inflated_confidence"},
                 "multiple_faults": {r + "_agent": "inflated_confidence" for r in ROLES[:3]}}
    output = {}
    for name, faults in scenarios.items():
        data = InsuranceAdjudication(InsuranceClaim(), faults=faults).export()
        output[name] = {"baseline": data["round"]["baseline"], "weighted": data["round"]["consensus"],
                        "proposed": data["final"]["decision"],
                        "assessments": {a["agent_id"]: a["assessment"] for a in data["agents"]}}
    return output
