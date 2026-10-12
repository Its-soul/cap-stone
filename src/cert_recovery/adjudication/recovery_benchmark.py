"""Selective recovery versus full recomputation benchmark and demonstration.

Compares selective recovery against full pipeline recomputation across four scenarios:
1. Single agent failure (risk_agent)
2. Single evidence fact change (evidence documents)
3. Multiple affected branches (policy coverage + evidence invoices)
4. Repeated changes before recovery (policy limit + compliance consent)

Tracks recomputed work, preserved versions, configured cost units, and measured runtime.
"""
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

from .domain import InsuranceClaim, ROLES
from .workflow import InsuranceAdjudication


def apply_claim_changes(workflow, claim: InsuranceClaim) -> bool:
    """Update claim fact dependencies in the engine without executing immediate recovery."""
    if claim.claim_id != workflow.claim.claim_id:
        raise ValueError("Cannot replace claim identity")
    changed = False
    for role in ROLES:
        source = workflow.system.latest("facts:" + role)
        if source.payload["facts"] != claim.facts(role):
            workflow.system.change_dependency(source.ref.artifact_id, {**source.payload, "facts": claim.facts(role)})
            changed = True
    workflow.claim = claim
    return changed


def run_single_comparison(setup_fn, recover_sel_fn, recover_full_fn, scenario_name: str, repetitions: int = 5) -> dict:
    """Run selective vs full recomputation comparison on identical initial state with repeated measurements."""
    sel_times = []
    full_times = []
    last_wf_sel = None
    last_wf_full = None
    last_outcome_sel = None
    last_outcome_full = None

    for _ in range(repetitions):
        # 1. Selective Recovery
        wf_sel, target_claim = setup_fn()
        t0 = time.perf_counter()
        outcome_sel = recover_sel_fn(wf_sel, target_claim)
        time_sel_ms = (time.perf_counter() - t0) * 1000
        sel_times.append(time_sel_ms)
        last_wf_sel = wf_sel
        last_outcome_sel = outcome_sel

        # 2. Full Recomputation (on identical initial state)
        wf_full, target_claim = setup_fn()
        t0 = time.perf_counter()
        outcome_full = recover_full_fn(wf_full, target_claim)
        time_full_ms = (time.perf_counter() - t0) * 1000
        full_times.append(time_full_ms)
        last_wf_full = wf_full
        last_outcome_full = outcome_full

    wf_sel = last_wf_sel
    wf_full = last_wf_full
    outcome_sel = last_outcome_sel
    outcome_full = last_outcome_full

    # Counts
    preserved_agents_sel = sum(wf_sel.system.latest(r + "_agent").ref.version == 1 for r in ROLES)
    preserved_agents_full = sum(wf_full.system.latest(r + "_agent").ref.version == 1 for r in ROLES)

    recomputed_agents_sel = sum(wf_sel.executions[r] - 1 for r in ROLES)
    recomputed_agents_full = sum(wf_full.executions[r] - 1 for r in ROLES)

    preserved_artifacts_sel = [
        {
            "artifact_id": r + "_agent",
            "version": wf_sel.system.latest(r + "_agent").ref.version,
            "content_hash": wf_sel.system.latest(r + "_agent").content_hash,
        }
        for r in ROLES if wf_sel.system.latest(r + "_agent").ref.version == 1
    ]

    sel_mean = sum(sel_times) / repetitions
    full_mean = sum(full_times) / repetitions

    return {
        "scenario": scenario_name,
        "selective": {
            "status": outcome_sel.status,
            "restored_artifacts_count": len(outcome_sel.restored),
            "restored_artifact_ids": [r.artifact_id for r in outcome_sel.restored],
            "recomputed_agent_executions": recomputed_agents_sel,
            "preserved_agent_versions_count": preserved_agents_sel,
            "preserved_artifacts": preserved_artifacts_sel,
            "total_executions_by_role": dict(wf_sel.executions),
            "configured_cost_units": outcome_sel.attempted_cost_units,
            "measured_runtime_ms": round(sel_mean, 3),
            "timing_distribution_ms": {
                "mean": round(sel_mean, 3),
                "min": round(min(sel_times), 3),
                "max": round(max(sel_times), 3),
                "repetitions": repetitions,
                "raw_samples": [round(t, 3) for t in sel_times],
            },
            "final_decision": wf_sel.result()["decision"],
            "is_valid": wf_sel.system.is_valid(wf_sel.system.latest("final").ref),
        },
        "full_recomputation": {
            "status": outcome_full.status,
            "restored_artifacts_count": len(outcome_full.restored),
            "restored_artifact_ids": [r.artifact_id for r in outcome_full.restored],
            "recomputed_agent_executions": recomputed_agents_full,
            "preserved_agent_versions_count": preserved_agents_full,
            "preserved_artifacts": [],
            "total_executions_by_role": dict(wf_full.executions),
            "configured_cost_units": outcome_full.attempted_cost_units,
            "measured_runtime_ms": round(full_mean, 3),
            "timing_distribution_ms": {
                "mean": round(full_mean, 3),
                "min": round(min(full_times), 3),
                "max": round(max(full_times), 3),
                "repetitions": repetitions,
                "raw_samples": [round(t, 3) for t in full_times],
            },
            "final_decision": wf_full.result()["decision"],
            "is_valid": wf_full.system.is_valid(wf_full.system.latest("final").ref),
        },
        "efficiency_gain": {
            "saved_agent_executions": recomputed_agents_full - recomputed_agents_sel,
            "saved_cost_units": outcome_full.attempted_cost_units - outcome_sel.attempted_cost_units,
            "speedup_ratio": round(full_mean / sel_mean, 2) if sel_mean > 0 else None,
            "speedup_ratio_min": round(min(full_times) / max(sel_times), 2) if max(sel_times) > 0 else None,
            "speedup_ratio_max": round(max(full_times) / min(sel_times), 2) if min(sel_times) > 0 else None,
            "is_slowdown": full_mean < sel_mean,
            "timing_note": "Mock agent execution in Python is near-instantaneous (~0.01ms); measured elapsed time reflects in-memory DAG traversal, cryptographic hashing, and Python interpreter overhead. Speedup is modest on mock backends and distributions can overlap.",
        },
    }


def full_recompute_recovery(workflow, target_claim=None):
    """Force full pipeline recomputation in a single pass without prior selective recovery."""
    if target_claim:
        apply_claim_changes(workflow, target_claim)
    for r in ROLES:
        workflow.system.invalidate_artifact(workflow.system.latest(r + "_agent").ref, "force full recomputation")
    return workflow._recover()


def run_recovery_benchmark() -> dict:
    """Execute the full 4-scenario recovery comparison benchmark."""
    scenarios = []

    # Scenario 1: Single agent failure (risk_agent)
    def setup_s1():
        c = InsuranceClaim()
        return InsuranceAdjudication(c, faults={"risk_agent": "failure"}), c

    def sel_s1(wf, _):
        return wf.recover_agent("risk_agent")

    def full_s1(wf, _):
        wf.faults.pop("risk_agent", None)
        return full_recompute_recovery(wf)

    scenarios.append(run_single_comparison(setup_s1, sel_s1, full_s1, "Single Agent Failure (risk_agent)"))

    # Scenario 2: Single evidence change (documents missing)
    def setup_s2():
        c = InsuranceClaim()
        return InsuranceAdjudication(c), replace(c, documents_available=False)

    def sel_s2(wf, new_c):
        return wf.update_claim(new_c)

    def full_s2(wf, new_c):
        return full_recompute_recovery(wf, new_c)

    scenarios.append(run_single_comparison(setup_s2, sel_s2, full_s2, "Single Fact Change (evidence documents)"))

    # Scenario 3: Multiple affected branches (policy coverage + evidence invoices)
    def setup_s3():
        c = InsuranceClaim()
        return InsuranceAdjudication(c), replace(c, hospitalization_covered=False, invoice_matches=False)

    def sel_s3(wf, new_c):
        return wf.update_claim(new_c)

    def full_s3(wf, new_c):
        return full_recompute_recovery(wf, new_c)

    scenarios.append(run_single_comparison(setup_s3, sel_s3, full_s3, "Multiple Affected Branches (policy + evidence)"))

    # Scenario 4: Repeated changes before recovery
    def setup_s4():
        c = InsuranceClaim()
        wf = InsuranceAdjudication(c)
        # Apply repeated dependency changes without immediate recovery
        s_pol = wf.system.latest("facts:policy")
        wf.system.change_dependency("facts:policy", {**s_pol.payload, "facts": {**s_pol.payload["facts"], "amount_inr": 80_000}})
        s_pol2 = wf.system.latest("facts:policy")
        wf.system.change_dependency("facts:policy", {**s_pol2.payload, "facts": {**s_pol2.payload["facts"], "amount_inr": 90_000}})
        s_comp = wf.system.latest("facts:compliance")
        wf.system.change_dependency("facts:compliance", {**s_comp.payload, "facts": {**s_comp.payload["facts"], "consent_available": False}})
        return wf, c

    def sel_s4(wf, _):
        return wf._recover()

    def full_s4(wf, _):
        return full_recompute_recovery(wf)

    scenarios.append(run_single_comparison(setup_s4, sel_s4, full_s4, "Repeated Changes Before Recovery (policy + compliance)"))

    return {
        "benchmark_version": "recovery-v1",
        "description": "Comparison of selective invalidation recovery vs full pipeline recomputation",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scenarios": scenarios,
    }


def save_recovery_benchmark(results: dict, output_path: str | Path) -> Path:
    """Save recovery benchmark results exclusively to prevent overwriting existing files."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, allow_nan=False)
    return path
