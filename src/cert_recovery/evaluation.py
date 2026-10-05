from __future__ import annotations

import copy
from dataclasses import asdict
from time import perf_counter

from .config import require_manual, resolve_path
from .data_pipeline import read_jsonl, write_json
from .engine import CertificateSystem
from .metrics import compare_plan_cost, ratio, recovery_savings
from .models import ArtifactKind as K, VerificationResult
from .provenance import AuditJournal

BASELINES = ("A_full_recomputation", "B_no_invalidation", "C_invalidation_only", "proposed")


def _root_value(nodes) -> int:
    roots = {node.ref.artifact_id: node for node in nodes if node.kind == K.DEPENDENCY}
    return sum(node.payload["value"] for node in roots.values())


def build_arithmetic_workflow(case: dict, costs: dict) -> tuple[CertificateSystem, list[str]]:
    """Prepared toy contract: each claim sums its declared source dependencies."""
    system = CertificateSystem(analysis_cost_units=costs["analysis_cost_units"])
    system.register_verifier("source-sum", lambda subject, ancestry: VerificationResult(
        subject.payload.get("value") == _root_value(ancestry), "claim must equal declared source sum"))
    decisions: list[str] = []
    previous_certificate = None
    for index, source in enumerate(case["sources"]):
        root = system.add_dependency(f"D{index}", {"value": source["value"]}, trusted=source.get("trusted", True))
        if not system.is_valid(root) or (case["topology"] == "cascade" and index and previous_certificate is None):
            previous_certificate = None
            continue
        evidence_id, claim_id, certificate_id, decision_id = (f"{prefix}{index}" for prefix in ("E", "Q", "C", "A"))
        evidence = system.add_artifact(evidence_id, K.EVIDENCE, {"value": source["value"]}, (root,),
                                       cost_units=costs["evidence_cost_units"])
        supports = (evidence, previous_certificate) if case["topology"] == "cascade" and index else (evidence,)
        ancestry = {node.ref: node for ref in supports for node in system.provenance(ref)}
        claim = system.add_artifact(claim_id, K.CLAIM, {"value": _root_value(ancestry.values())}, supports,
                                    cost_units=costs["claim_cost_units"])
        certificate = system.issue_certificate(certificate_id, claim, "source-sum",
                                                cost_units=costs["certificate_cost_units"])
        system.add_artifact(decision_id, K.DECISION, system.artifact(claim).payload, (certificate,),
                             cost_units=costs["decision_cost_units"])
        def rebuild_sum(previous, current_supports):
            ancestry = {node.ref: node for parent in current_supports for node in system.provenance(parent.ref)}
            return {"value": _root_value(ancestry.values())}
        def rebuild_decision(previous, current_supports):
            cert = system.certificate(current_supports[0].ref)
            return system.artifact(cert.subject).payload
        system.register_rebuilder(evidence_id, rebuild_sum)
        system.register_rebuilder(claim_id, rebuild_sum)
        system.register_rebuilder(decision_id, rebuild_decision)
        previous_certificate = certificate
        decisions.append(decision_id)
    return system, decisions


def validate_case(case: dict) -> None:
    if not isinstance(case.get("world_id"), str) or not case["world_id"]:
        raise ValueError("Workflow needs world_id")
    if case.get("topology") not in {"independent", "cascade"}:
        raise ValueError("Supported toy topologies: independent, cascade")
    sources = case.get("sources")
    if not isinstance(sources, list) or not sources or any(
        type(item.get("value")) is not int or type(item.get("trusted", True)) is not bool for item in sources
    ):
        raise ValueError("Sources require integer values and Boolean trust")
    change = case.get("change", {})
    if (type(change.get("source_index")) is not int or not 0 <= change["source_index"] < len(sources)
            or change.get("kind") not in {"update", "remove", "untrusted"}
            or type(change.get("value", 0)) is not int):
        raise ValueError("Invalid source change")


def _oracle(case: dict, index: int) -> int | None:
    indices = range(index + 1) if case["topology"] == "cascade" else (index,)
    if any(not case["sources"][item].get("trusted", True) for item in indices):
        return None
    return sum(case["sources"][item]["value"] for item in indices)


def run_workflow_case(case: dict, baseline: str, config: dict, *, manual: bool = False) -> dict:
    require_manual(config, "allow_research_experiments", manual)
    validate_case(case)
    if baseline not in BASELINES:
        raise ValueError("Unknown baseline")
    complete_start = perf_counter()
    system, decisions = build_arithmetic_workflow(case, config["recovery"])
    setup_seconds = perf_counter() - complete_start
    original_nodes = system.current_artifacts()
    initial_certificates = [node for node in original_nodes if node.kind == K.CERTIFICATE]
    change = case["change"]
    updated = copy.deepcopy(case)
    source = updated["sources"][change["source_index"]]
    source["value"] = change.get("value", source["value"])
    source["trusted"] = change["kind"] == "update"
    dependency_id = f"D{change['source_index']}"
    start = perf_counter()
    report, outcome, costs = None, None, None
    if baseline == "A_full_recomputation":
        system, decisions = build_arithmetic_workflow(updated, config["recovery"])
        work_units = sum(node.cost_units for node in system.current_artifacts())
    elif baseline == "B_no_invalidation":
        # Deliberately stale control: sources change outside the cached certificate system.
        # The source oracle below, not this old graph, evaluates freshness/correctness.
        work_units = 0.0
    else:
        report = (system.remove_dependency(dependency_id) if change["kind"] == "remove" else
                  system.change_dependency(dependency_id, {"value": source["value"]}, trusted=source["trusted"]))
        work_units = config["recovery"]["analysis_cost_units"]
        if baseline == "proposed":
            plan = system.plan_recovery(report, config["recovery"]["budget_units"])
            costs = asdict(compare_plan_cost(plan, system.current_artifacts()))
            outcome = system.execute_recovery(plan)
            work_units = outcome.attempted_cost_units
    records = []
    for decision_id in decisions:
        index = int(decision_id[1:])
        node = system.latest(decision_id)
        if system.is_valid(node.ref):
            payload = system.commit_decision(node.ref)
            expected = _oracle(updated, index)
            touched = change["source_index"] <= index if case["topology"] == "cascade" else change["source_index"] == index
            fresh = baseline != "B_no_invalidation" or not touched
            records.append({"decision_id": decision_id, "committed": True,
                            "safe": fresh and expected is not None and payload["value"] == expected})
        else:
            records.append({"decision_id": decision_id, "committed": False, "safe": None})
    elapsed = perf_counter() - start
    complete_seconds = perf_counter() - complete_start
    committed = sum(row["committed"] for row in records)
    safe = sum(row["safe"] is True for row in records)
    expected_affected = {f"C{index}" for index in range(len(case["sources"]))
                         if (change["source_index"] <= index if case["topology"] == "cascade"
                             else change["source_index"] == index)}
    actual_affected = {ref.artifact_id for ref in report.certificates} if report else set()
    restored_certificates = sum(system.artifact(ref).kind == K.CERTIFICATE for ref in outcome.restored) if outcome else 0
    verification_units = sum(system.artifact(ref).cost_units for ref in outcome.attempted
                             if system.artifact(ref).kind == K.CERTIFICATE) if outcome else 0.0
    return {"world_id": case["world_id"], "baseline": baseline, "status": "RUN_MANUALLY",
            "scope": "toy arithmetic workflow; not external model validation",
            "safe_commits": safe, "total_commits": committed, "total_tasks": len(case["sources"]),
            "support_valid_commit_rate": ratio(safe, committed),
            "useful_completion_rate": ratio(safe, len(case["sources"])),
            "latency_seconds": elapsed, "configured_work_units": work_units,
            "initial_setup_seconds": setup_seconds, "total_elapsed_seconds": complete_seconds,
            "timing_scope": "local CPU wall time including change, policy, verification, commits; total includes fresh initial setup",
            "blocked_tasks": len(case["sources"]) - committed,
            "initial_artifacts": len(original_nodes), "final_artifacts": len(system.current_artifacts()),
            "audit_events": len(system.audit_export()),
            "impact": asdict(report) if report else None, "recovery": asdict(outcome) if outcome else None,
            "planned_cost": costs, "decisions": records,
            "declared_impact_precision": ratio(len(actual_affected & expected_affected), len(actual_affected)) if report else None,
            "declared_impact_recall": ratio(len(actual_affected & expected_affected), len(expected_affected)) if report else None,
            "invalidated_certificate_ratio": ratio(len(actual_affected), len(initial_certificates)) if report else None,
            "recovery_success_rate": ratio(restored_certificates, len(actual_affected)) if outcome else None,
            "verification_work_units": verification_units,
            "recomputation_ratio": ratio(len(outcome.restored), sum(node.kind not in {K.DEPENDENCY, K.VERIFICATION}
                                                                      for node in original_nodes)) if outcome else None,
            "provenance_bytes": len(__import__("json").dumps(system.audit_export()).encode("utf-8"))}


def paired_cost_records(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int | None], dict[str, dict]] = {}
    for record in records:
        grouped.setdefault((record["world_id"], record.get("repetition")), {})[record["baseline"]] = record
    result = []
    for (world, repetition), methods in grouped.items():
        if "proposed" in methods and "A_full_recomputation" in methods:
            proposed, full = methods["proposed"], methods["A_full_recomputation"]
            result.append({"world_id": world, "repetition": repetition, "configured_work_savings": recovery_savings(
                proposed["configured_work_units"], full["configured_work_units"]),
                "proposed_useful_completion": proposed["useful_completion_rate"],
                "full_useful_completion": full["useful_completion_rate"],
                "warning": "Compare savings only with useful completion and support-validity; blocked work is cheaper."})
    return result


def run_prepared_benchmark(config: dict, *, manual: bool = False) -> list[dict]:
    require_manual(config, "allow_research_experiments", manual)
    cases = read_jsonl(resolve_path(config, "workflow_cases"))
    if not cases or len({case["world_id"] for case in cases}) != len(cases):
        raise ValueError("Cases must have unique nonempty world IDs")
    records = [run_workflow_case(case, baseline, config, manual=True)
               for case in cases for baseline in config["evaluation"]["baselines"]]
    write_json(resolve_path(config, "results") / "workflow_benchmark.json",
               {"status": "RUN_MANUALLY", "records": records, "paired_cost": paired_cost_records(records)})
    return records


def run_integrity_experiment(config: dict, *, manual: bool = False) -> dict:
    require_manual(config, "allow_research_experiments", manual)
    journal = AuditJournal()
    journal.append("source", {"id": "D", "value": 1})
    journal.append("certificate", {"subject": "D@1", "result": "passed"})
    records, checkpoint = journal.export(), journal.checkpoint()
    edited = copy.deepcopy(records)
    edited[0]["payload"]["value"] = 2
    outcomes = {"clean": AuditJournal.verify(records, checkpoint),
                "edit_detected": not AuditJournal.verify(edited, checkpoint),
                "reorder_detected": not AuditJournal.verify(list(reversed(records)), checkpoint),
                "truncation_detected_with_checkpoint": not AuditJournal.verify(records[:-1], checkpoint),
                "truncation_undetectable_without_checkpoint": AuditJournal.verify(records[:-1])}
    write_json(resolve_path(config, "results") / "integrity_experiment.json", {"status": "RUN_MANUALLY", "checks": outcomes})
    return outcomes


def run_direct_invalidation_experiment(config: dict, *, manual: bool = False) -> dict:
    require_manual(config, "allow_research_experiments", manual)
    system = CertificateSystem()
    system.register_verifier("nonnegative-source", lambda subject, ancestry: VerificationResult(
        subject.payload["value"] >= 0, "nonnegative source value required"))
    dependency = system.add_dependency("D42", {"value": 1})
    certificate = system.issue_certificate("C-direct", dependency, "nonnegative-source")
    report = system.change_dependency("D42", {"value": 2})
    direct_ids = [ref.artifact_id for ref in report.direct_certificates]
    output = {"status": "RUN_MANUALLY", "expected_direct_ids": ["C-direct"],
              "observed_direct_ids": direct_ids, "old_state": system.certificate_status(certificate).value,
              "historical_artifact_hash_valid": system.artifact(certificate).integrity_valid(),
              "impact": asdict(report)}
    write_json(resolve_path(config, "results") / "direct_invalidation.json", output)
    return output


def run_replacement_experiment(config: dict, *, manual: bool = False) -> dict:
    require_manual(config, "allow_research_experiments", manual)
    case = {"world_id": "replacement-mechanics", "topology": "independent", "sources": [{"value": 1}],
            "change": {"source_index": 0, "kind": "remove"}}
    system, _ = build_arithmetic_workflow(case, config["recovery"])
    failure = system.remove_dependency("D0")
    blocked = system.execute_recovery(system.plan_recovery(failure))
    system.change_dependency("D0", {"value": 8}, trusted=True)
    restored = system.execute_recovery(system.plan_pending_recovery())
    output = {"status": "RUN_MANUALLY", "while_unavailable": asdict(blocked),
              "after_replacement": asdict(restored), "decision_can_commit": system.is_valid(system.latest("A0").ref)}
    write_json(resolve_path(config, "results") / "replacement_experiment.json", output)
    return output


def run_fault_experiment(config: dict, *, manual: bool = False) -> list[dict]:
    require_manual(config, "allow_research_experiments", manual)
    case = {"world_id": "fault-mechanics", "topology": "independent", "sources": [{"value": 1}],
            "change": {"source_index": 0, "kind": "update", "value": 2}}
    output = []
    for fault in ("missing_recipe", "stale_plan", "budget_exhaustion"):
        system, _ = build_arithmetic_workflow(case, config["recovery"])
        if fault == "missing_recipe":
            # This explicit fixture edits a trusted registry; never expose this as an agent tool.
            system._recipes.pop("Q0")
        report = system.change_dependency("D0", {"value": 2})
        plan = system.plan_recovery(report, budget_units=0 if fault == "budget_exhaustion" else None)
        if fault == "stale_plan":
            system.change_dependency("D0", {"value": 3})
        try:
            outcome = asdict(system.execute_recovery(plan))
        except ValueError as error:
            outcome = {"rejected": str(error)}
        output.append({"fault": fault, "outcome": outcome,
                       "decision_can_commit": system.is_valid(system.latest("A0").ref)})
    write_json(resolve_path(config, "results") / "fault_experiment.json", {"status": "RUN_MANUALLY", "records": output})
    return output
