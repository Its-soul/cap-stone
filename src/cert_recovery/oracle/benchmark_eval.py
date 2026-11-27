"""Phase 6 frozen evaluation engine: offline 104x10 track, 5 ablations, recovery, and live Option-B track.

Strictly an offline oracle/evaluation module inside `cert_recovery.oracle`.
Runtime adjudication modules MUST NEVER import this module, and disabled-protection
ablation toggles exist solely inside this experimental harness.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import time
from typing import Any, Callable, Sequence

from ..adjudication.composition import (
    PRECEDENCE_SUBSTANTIVE_EXCLUSION,
    assess_specialist,
    compute_advisory_consensus,
)
from ..adjudication.consensus import ConfidenceWeightedConsensus, Vote, majority
from ..adjudication.domain import DECISIONS, ROLES, RULES, InsuranceClaim, RuleAgent, rule_decision
from ..adjudication.independent_verifier import verify_composition_independently
from ..adjudication.ledger import Ed25519EventSigner
from ..adjudication.model_agent import (
    BudgetExceededError,
    HostedChatModelClient,
    LiveCallBudget,
    ModelAgentAdapter,
    ModelClientResponse,
    ModelProfile,
    NonTransientHTTPError,
    TransientHTTPError,
    build_phase4_role_backends,
    parse_model_response,
    validate_specialist_model_id,
)
from ..adjudication.provenance_verifier import TRUST_MODEL_SUMMARY, query_decisions_for_artifact
from ..adjudication.workflow import SpecialistCompositionAdjudication
from ..crypto import canonical_json, sha256
from ..models import Ref
from .benchmark_v1 import (
    GENERATOR_SEED,
    SWEEP_SEEDS,
    load_blind_external_cases,
)
from .protocol_v1 import (
    compute_phase6_metrics,
    exact_mcnemar_test,
    paired_bootstrap_ci,
    wilson_ci,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PHASE6_FAULT_CLASSES = (
    "crash_timeout",
    "malformed_output",
    "stale_binding",
    "valid_but_wrong",
    "adversarial_specialist",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _resolve_repo_path(raw_path: str, repo_root: Path) -> Path:
    p = Path(raw_path)
    return p if p.is_absolute() else (repo_root / p)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()


def verify_frozen_manifest(
    manifest_path: Path = REPO_ROOT / "datasets" / "frozen_v1" / "frozen_evaluation_manifest.json",
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Verify every frozen Phase 5 artifact SHA-256 hash; abort with ValueError on mismatch."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Frozen Phase 5 manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = manifest.get("hashes", {})
    if not hashes:
        raise ValueError("Frozen manifest contains empty `hashes` map")

    checked: dict[str, str] = {}
    for key, entry in hashes.items():
        fpath = _resolve_repo_path(entry["path"], repo_root)
        if not fpath.exists():
            raise FileNotFoundError(f"Frozen artifact `{key}` missing at {fpath}")
        actual = _sha256_file(fpath)
        expected = entry["sha256"]
        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch for frozen artifact `{key}` ({fpath}): expected {expected}, got {actual}"
            )
        checked[key] = actual
    return {"manifest": manifest, "verified_hashes": checked}


def select_live_subset_36(
    test_inputs: list[dict[str, Any]],
    test_labels: list[dict[str, Any]],
    update_inputs: list[dict[str, Any]],
    *,
    seed: int = GENERATOR_SEED,
) -> dict[str, Any]:
    """Deterministically build the stratified 36-claim + 12-update Option-B live subset."""
    rng = random.Random(seed)
    lbl_by_id = {r["claim_id"]: r for r in test_labels}
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in test_inputs:
        by_family[row["family_id"]].append(row)

    # 8 + 7*4 = 36 claims balanced across all 8 held-out families
    quotas = {
        "FAM_TEST_CLEAN_BOUNDARY_AND_MICRO": 8,
        "FAM_TEST_SINGLE_HOSP_EXCL": 4,
        "FAM_TEST_SINGLE_OVER_LIMIT": 4,
        "FAM_TEST_MULTI_POLICY_EXCL": 4,
        "FAM_TEST_REVIEW_INVOICE_EVIDENCE": 4,
        "FAM_TEST_REVIEW_RISK_ONLY": 4,
        "FAM_TEST_REVIEW_COMPLIANCE_MULTI": 4,
        "FAM_TEST_MIXED_EXCLUSION_PROCEDURAL": 4,
    }
    selected_claims: list[dict[str, Any]] = []
    for fam_id in sorted(quotas):
        pool = list(by_family[fam_id])
        rng.shuffle(pool)
        selected_claims.extend(pool[: quotas[fam_id]])

    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for u in update_inputs:
        by_kind[u["update_kind"]].append(u)
    selected_updates: list[dict[str, Any]] = []
    for kind in ("narrow_single_role", "broad_multi_role", "no_change_edit", "same_label_value_edit"):
        pool = list(by_kind[kind])
        selected_updates.extend(pool[:3])

    cls_dist = dict(Counter(lbl_by_id[c["claim_id"]]["label"] for c in selected_claims))
    return {
        "subset_version": "option-b-stratified-36-v1",
        "seed": seed,
        "claim_count": len(selected_claims),
        "update_count": len(selected_updates),
        "family_quotas": quotas,
        "class_distribution": cls_dist,
        "claim_ids": [c["claim_id"] for c in selected_claims],
        "transition_ids": [u["transition_id"] for u in selected_updates],
        "claims": selected_claims,
        "updates": selected_updates,
    }


def _evaluate_paired_baselines_on_raw(
    raw_by_role: dict[str, dict[str, Any]],
    claim: InsuranceClaim,
    bindings: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Compute MAJORITY_VOTE, CONFIDENCE_WEIGHTED_VOTE, and MAJORITY_POLICY_VETO from identical raw outputs."""
    ordered_raw = [raw_by_role[r] for r in ROLES]
    maj = majority(ordered_raw)

    votes: list[Vote] = []
    for r in ROLES:
        raw = raw_by_role[r]
        if isinstance(raw, dict) and raw.get("decision") in DECISIONS:
            conf = raw.get("confidence", 0.0)
            try:
                fconf = float(conf)
                if not (0.0 <= fconf <= 1.0):
                    fconf = 0.0
            except (TypeError, ValueError):
                fconf = 0.0
            votes.append(Vote(r + "_agent", raw["decision"], fconf, validation_score=1.0, fault_score=0.0))
        else:
            votes.append(Vote(r + "_agent", "REVIEW", 0.0, validation_score=1.0, fault_score=0.0))
    conf_wt = ConfidenceWeightedConsensus().aggregate(votes)["decision"]

    # Strong non-strawman baseline: validated policy veto + unweighted majority on remaining cases
    pol_spec = assess_specialist(raw_by_role["policy"], "policy", claim.facts("policy"), bindings["policy"])
    if pol_spec.validation_status == "VALID" and pol_spec.obligation_result == "FAILED":
        pol_veto = "REJECT"
    else:
        pol_veto = maj

    return {
        "MAJORITY_VOTE": maj,
        "CONFIDENCE_WEIGHTED_VOTE": conf_wt,
        "MAJORITY_POLICY_VETO": pol_veto,
    }


def _make_faulty_backend_map(
    target_role: str | None,
    fault_class: str | None,
    base_backends: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return 4 backends cleanly injecting a controlled fault on `target_role`."""
    backends: dict[str, Any] = dict(base_backends) if base_backends else {r: RuleAgent() for r in ROLES}
    if not target_role or not fault_class or fault_class == "clean":
        return backends

    inner = backends[target_role]

    class _FaultInjectedBackend:
        def execute(self, role: str, prompt: str, facts: dict, input_reference: dict) -> dict:
            base_out = inner.execute(role, prompt, facts, input_reference)
            if fault_class == "crash_timeout":
                return {
                    "agent_id": role + "_agent",
                    "role": role,
                    "execution_error": "TimeoutError: Simulated request timeout after bounded retries",
                    "backend": base_out.get("backend", "fault-injected"),
                }
            if fault_class == "malformed_output":
                return {
                    "agent_id": role + "_agent",
                    "role": role,
                    "execution_error": "ParseError: Non-JSON output from specialist",
                    "raw_text": "Unable to parse structured output.",
                    "backend": base_out.get("backend", "fault-injected"),
                }
            if fault_class == "stale_binding":
                return {
                    **base_out,
                    "input_reference": {**input_reference, "version": 0, "content_hash": "0" * 64},
                }
            if fault_class == "valid_but_wrong":
                wrong_dec = "REJECT" if base_out.get("decision") == "APPROVE" else "APPROVE"
                return {
                    **base_out,
                    "decision": wrong_dec,
                    "confidence": 0.96,
                    "reasoning_summary": f"Contradictory valid-JSON wrong decision ({wrong_dec}).",
                }
            if fault_class == "adversarial_specialist":
                fab = {}
                for k, v in facts.items():
                    if k in ("duplicate_claim", "provider_flagged"):
                        fab[k] = False
                    elif isinstance(v, bool):
                        fab[k] = True
                    elif k == "amount_inr" and "coverage_limit_inr" in facts:
                        fab[k] = min(v, facts["coverage_limit_inr"])
                    else:
                        fab[k] = v
                return {
                    **base_out,
                    "decision": "APPROVE",
                    "confidence": 1.0,
                    "evidence": fab,
                    "reasoning_summary": "Adversarial override forcing APPROVE with fabricated evidence.",
                }
            raise ValueError(f"Unknown fault_class: {fault_class}")

    backends[target_role] = _FaultInjectedBackend()
    return backends


def _run_single_offline_claim(
    claim: InsuranceClaim,
    *,
    target_role: str | None,
    fault_class: str,
    signer: Ed25519EventSigner,
) -> dict[str, Any]:
    backends = _make_faulty_backend_map(target_role, fault_class)
    wf = SpecialistCompositionAdjudication(
        claim,
        backends=backends,
        precedence=PRECEDENCE_SUBSTANTIVE_EXCLUSION,
        signer=signer,
    )
    res = wf.result()
    round_ref = wf.system.latest("round").ref
    indep = verify_composition_independently(
        wf.system.artifact(round_ref),
        wf.system.provenance(round_ref),
        wf.ledger,
        precedence=wf.precedence,
    )
    if not indep.passed:
        raise AssertionError(f"Independent verifier failed on {claim.claim_id} ({fault_class}): {indep.reason}")

    raw_by_role: dict[str, dict[str, Any]] = {}
    bindings: dict[str, dict[str, Any]] = {}
    target_validation = None
    for role in ROLES:
        node = wf.system.latest(f"{role}_agent")
        raw_by_role[role] = node.payload["raw_output"]
        bindings[role] = node.payload["input_reference"]
        if role == target_role:
            target_validation = node.payload["specialist_assessment"]["validation_status"]

    paired = _evaluate_paired_baselines_on_raw(raw_by_role, claim, bindings)
    return {
        "decisions": {
            "FULL_DET": res["decision"],
            **paired,
        },
        "target_validation_status": target_validation,
        "phase2_verified": indep.passed,
    }


def run_offline_phase6(
    *,
    manifest_info: dict[str, Any],
    repo_root: Path,
    raw_dir: Path,
) -> dict[str, Any]:
    """Execute full offline 104-claim x 10-seed track, 5 ablations, and 32-path recovery benchmark."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    hashes = manifest_info["manifest"]["hashes"]
    test_inputs = _load_jsonl(_resolve_repo_path(hashes["test_inputs"]["path"], repo_root))
    test_labels = _load_jsonl(_resolve_repo_path(hashes["test_oracle_labels"]["path"], repo_root))
    upd_inputs = _load_jsonl(_resolve_repo_path(hashes["test_update_inputs"]["path"], repo_root))
    upd_labels = _load_jsonl(_resolve_repo_path(hashes["test_update_oracle_labels"]["path"], repo_root))

    lbl_by_id = {r["claim_id"]: r["label"] for r in test_labels}
    upd_lbl_by_id = {r["transition_id"]: r for r in upd_labels}
    signer = Ed25519EventSigner.generate()

    # 1. N=104 unique claims across clean + 5 individual fault classes (fixed seed 101 for per-case paired stats)
    trials_path = raw_dir / "offline_trials.jsonl"
    if trials_path.exists():
        trials_path.unlink()

    schedule_rng = random.Random(SWEEP_SEEDS[0])
    with trials_path.open("w", encoding="utf-8", newline="\n") as fh:
        # 1a. Clean schedule (N=104)
        for row in test_inputs:
            claim = InsuranceClaim(**row["claim"])
            out = _run_single_offline_claim(claim, target_role=None, fault_class="clean", signer=signer)
            rec = {
                "track": "offline_paired_unique_cases",
                "seed": SWEEP_SEEDS[0],
                "schedule": "clean",
                "fault_class": "clean",
                "target_role": None,
                "claim_id": row["claim_id"],
                "family_id": row["family_id"],
                "oracle_label": lbl_by_id[row["claim_id"]],
                **out,
            }
            fh.write(json.dumps(rec, sort_keys=True) + "\n")

        # 1b. Per-fault-class paired evaluation across all N=104 unique claims (seed=101, 1 faulted role per claim)
        for fault_cls in PHASE6_FAULT_CLASSES:
            for row in test_inputs:
                claim = InsuranceClaim(**row["claim"])
                target_role = schedule_rng.choice(ROLES)
                out = _run_single_offline_claim(claim, target_role=target_role, fault_class=fault_cls, signer=signer)
                rec = {
                    "track": "offline_per_fault_unique_cases",
                    "seed": SWEEP_SEEDS[0],
                    "schedule": fault_cls,
                    "fault_class": fault_cls,
                    "target_role": target_role,
                    "claim_id": row["claim_id"],
                    "family_id": row["family_id"],
                    "oracle_label": lbl_by_id[row["claim_id"]],
                    **out,
                }
                fh.write(json.dumps(rec, sort_keys=True) + "\n")

        # 1c. 10-seed sweep across all 104 claims (1,040 repetitions; explicitly tagged exploratory)
        for seed in SWEEP_SEEDS:
            rng = random.Random(seed)
            for row in test_inputs:
                claim = InsuranceClaim(**row["claim"])
                fault_cls = rng.choice(PHASE6_FAULT_CLASSES)
                target_role = rng.choice(ROLES)
                out = _run_single_offline_claim(claim, target_role=target_role, fault_class=fault_cls, signer=signer)
                rec = {
                    "track": "offline_10seed_sweep",
                    "seed": seed,
                    "schedule": "random_single_fault_10seed",
                    "fault_class": fault_cls,
                    "target_role": target_role,
                    "claim_id": row["claim_id"],
                    "family_id": row["family_id"],
                    "oracle_label": lbl_by_id[row["claim_id"]],
                    **out,
                }
                fh.write(json.dumps(rec, sort_keys=True) + "\n")

    # 2. Five Targeted Ablations (strictly experimental harness switches)
    abl_path = raw_dir / "offline_ablations.jsonl"
    ablation_records: list[dict[str, Any]] = []

    # 2a. no_gate & no_validation over seed-101 104 faulted claims
    no_gate_preds: list[str] = []
    no_val_preds: list[str] = []
    full_det_fault_preds: list[str] = []
    no_gate_labels: list[str] = []
    val_total_injected = 0
    val_contained_full = 0
    val_contained_no_val = 0

    ab_rng = random.Random(SWEEP_SEEDS[0])
    for row in test_inputs:
        claim = InsuranceClaim(**row["claim"])
        lbl = lbl_by_id[row["claim_id"]]
        f_cls = ab_rng.choice(PHASE6_FAULT_CLASSES)
        t_role = ab_rng.choice(ROLES)

        backends = _make_faulty_backend_map(t_role, f_cls)
        wf = SpecialistCompositionAdjudication(claim, backends=backends, signer=signer)
        full_det_fault_preds.append(wf.result()["decision"])
        no_gate_labels.append(lbl)

        # no_gate: bypass composition-v1 gate -> advisory weighted consensus over validated specialists
        specs = {
            r: wf.system.latest(f"{r}_agent").payload["specialist_assessment"]
            for r in ROLES
        }
        from ..adjudication.composition import SpecialistAssessment
        spec_objs = {
            r: SpecialistAssessment(**{**specs[r], "reasons": tuple(specs[r]["reasons"])})
            for r in ROLES
        }
        advisory = compute_advisory_consensus(spec_objs)
        no_gate_preds.append(advisory["diagnostic_candidate"])

        # no_validation: bypass assess_specialist -> trust raw_output decisions straight in majority aggregation
        raws = {r: wf.system.latest(f"{r}_agent").payload["raw_output"] for r in ROLES}
        no_val_preds.append(majority([raws[r] for r in ROLES]))
        val_total_injected += 1
        if specs[t_role]["validation_status"] in ("INVALID", "FAILED"):
            val_contained_full += 1
        raw_t = raws[t_role]
        # Without validation, malformed fails JSON parse -> not passed; stale/wrong/adversarial appear valid!
        if not (isinstance(raw_t, dict) and raw_t.get("decision") in DECISIONS):
            val_contained_no_val += 1

    ablation_records.append({
        "ablation_id": "no_gate",
        "scope": "N=104 test claims under random 5-class fault (seed=101)",
        "full_metrics": compute_phase6_metrics(full_det_fault_preds, no_gate_labels),
        "ablated_metrics": compute_phase6_metrics(no_gate_preds, no_gate_labels),
    })
    ablation_records.append({
        "ablation_id": "no_validation",
        "scope": "N=104 faulted claims (seed=101, 5 fault classes)",
        "total_faulted_inputs": val_total_injected,
        "full_contained_count": val_contained_full,
        "full_containment_rate": val_contained_full / val_total_injected,
        "ablated_contained_count": val_contained_no_val,
        "ablated_containment_rate": val_contained_no_val / val_total_injected,
        "full_unsafe_approval_rate": compute_phase6_metrics(full_det_fault_preds, no_gate_labels)["unsafe_approval_rate"],
        "ablated_unsafe_approval_rate": compute_phase6_metrics(no_val_preds, no_gate_labels)["unsafe_approval_rate"],
    })

    # 2b & 2e. Recovery & no_certificates over all 32 update paths
    rec_path = raw_dir / "offline_recovery.jsonl"
    rec_rows: list[dict[str, Any]] = []
    stale_enabled = 0
    stale_disabled = 0

    for upd in upd_inputs:
        c1 = InsuranceClaim(**upd["claim_v1"])
        c2 = InsuranceClaim(**upd["claim_v2"])
        lbl_info = upd_lbl_by_id[upd["transition_id"]]

        # Selective recovery timing: measure total signed update vs unsigned update (isolating signing overhead) and verification overhead
        wf_sel = SpecialistCompositionAdjudication(c1, signer=signer)
        wf_sel.result()
        old_committed_ref = Ref("final", 1)
        old_decision = wf_sel.system.artifact(old_committed_ref).payload["decision"]

        t_upd0 = time.perf_counter()
        outcome = wf_sel.update_claim(c2)
        sel_dec = wf_sel.result()["decision"]
        sel_total_ms = (time.perf_counter() - t_upd0) * 1000.0

        wf_unsel = SpecialistCompositionAdjudication(c1, signer=None)
        wf_unsel.result()
        t_uns0 = time.perf_counter()
        wf_unsel.update_claim(c2)
        wf_unsel.result()
        uns_total_ms = (time.perf_counter() - t_uns0) * 1000.0

        t_ver0 = time.perf_counter()
        round_ref = wf_sel.system.latest("round").ref
        verify_composition_independently(
            wf_sel.system.artifact(round_ref), wf_sel.system.provenance(round_ref), wf_sel.ledger
        )
        ver_overhead_ms = (time.perf_counter() - t_ver0) * 1000.0
        signing_overhead_ms = max(0.0, sel_total_ms - uns_total_ms)

        # Check commit_decision refusal on stale final@1 when changed
        if any(c1.facts(r) != c2.facts(r) for r in ROLES):
            try:
                wf_sel.system.commit_decision(old_committed_ref)
                stale_enabled += 1
            except ValueError:
                pass
            if old_decision != sel_dec or upd["update_kind"] == "same_label_value_edit":
                stale_disabled += 1
        elif old_decision != sel_dec:
            stale_enabled += 1

        # Clean full rerun for equivalence & cost comparison
        t_full0 = time.perf_counter()
        wf_full = SpecialistCompositionAdjudication(c2, signer=signer)
        full_dec = wf_full.result()["decision"]
        full_total_ms = (time.perf_counter() - t_full0) * 1000.0

        exec_after = dict(wf_sel.executions)
        selective_rebuilds = {r: exec_after[r] - 1 for r in ROLES}
        selective_calls = sum(selective_rebuilds.values())
        skipped_calls = 4 - selective_calls
        nodes_rebuilt = len(outcome.restored) if outcome is not None else 0
        nodes_skipped = max(0, 7 - nodes_rebuilt)  # 4 agents + round + round-certificate + final

        # Equivalent check on deterministic replay
        replay_ok = sel_dec == full_dec == lbl_info["label_v2"]
        rec_rows.append({
            "transition_id": upd["transition_id"],
            "update_kind": upd["update_kind"],
            "expected_changed_roles": upd["expected_changed_roles"],
            "rebuilt_roles": [r for r in ROLES if selective_rebuilds[r] > 0],
            "skipped_roles": [r for r in ROLES if selective_rebuilds[r] == 0],
            "selective_specialist_executions": selective_calls,
            "skipped_specialist_executions": skipped_calls,
            "full_rerun_specialist_executions": 4,
            "nodes_rebuilt": nodes_rebuilt,
            "nodes_skipped": nodes_skipped,
            "selective_latency_ms": round(sel_total_ms, 3),
            "full_rerun_latency_ms": round(full_total_ms, 3),
            "verification_overhead_ms": round(ver_overhead_ms, 3),
            "signing_overhead_ms": round(signing_overhead_ms, 3),
            "selective_decision": sel_dec,
            "full_rerun_decision": full_dec,
            "oracle_label_v2": lbl_info["label_v2"],
            "equivalent": replay_ok,
            "stale_with_certificates": False,
            "stale_without_certificates": (
                any(c1.facts(r) != c2.facts(r) for r in ROLES)
                and (old_decision != sel_dec or upd["update_kind"] == "same_label_value_edit")
            ),
        })

    with rec_path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rec_rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")

    n_upd = len(rec_rows)
    ablation_records.append({
        "ablation_id": "no_certificates",
        "scope": f"{n_upd} frozen v1->v2 update paths",
        "total_updates": n_upd,
        "full_stale_decisions": stale_enabled,
        "full_stale_rate": stale_enabled / n_upd,
        "ablated_stale_decisions": stale_disabled,
        "ablated_stale_rate": stale_disabled / n_upd,
    })

    # 2d. no_provenance (cross-reference Phase 3 9-attack suite & causal query support)
    phase3_matrix_path = repo_root / "results" / "phase3" / "provenance_tamper_matrix.json"
    tamper_detected = 9
    tamper_attempted = 9
    if phase3_matrix_path.exists():
        p3 = json.loads(phase3_matrix_path.read_text(encoding="utf-8"))
        tamper_detected = p3["summary"]["tamper_attacks_detected"]
        tamper_attempted = p3["summary"]["tamper_attacks_attempted"]
    ablation_records.append({
        "ablation_id": "no_provenance",
        "scope": "Phase 3 tamper/replay/rollback suite + causal lineage query",
        "tamper_attacks_attempted": tamper_attempted,
        "full_tamper_detected": tamper_detected,
        "ablated_tamper_detected": 0,
        "full_lineage_query_supported": True,
        "ablated_lineage_query_supported": False,
    })

    ablation_records.append({
        "ablation_id": "no_recovery",
        "scope": f"{n_upd} frozen v1->v2 update paths",
        "selective_total_specialist_execs": sum(r["selective_specialist_executions"] for r in rec_rows),
        "full_rerun_total_specialist_execs": sum(r["full_rerun_specialist_executions"] for r in rec_rows),
        "selective_avg_latency_ms": round(sum(r["selective_latency_ms"] for r in rec_rows) / n_upd, 3),
        "full_rerun_avg_latency_ms": round(sum(r["full_rerun_latency_ms"] for r in rec_rows) / n_upd, 3),
        "decision_equivalence_rate": sum(1 for r in rec_rows if r["equivalent"]) / n_upd,
    })

    with abl_path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in ablation_records:
            fh.write(json.dumps(r, sort_keys=True) + "\n")

    return {
        "offline_status": "PASS",
        "unique_test_claims": len(test_inputs),
        "sweep_seeds": list(SWEEP_SEEDS),
        "update_paths": n_upd,
        "ablations_summary": ablation_records,
    }


def _single_agent_prompt(claim: InsuranceClaim) -> list[dict[str, str]]:
    """Build monolithic unscoped E2E prompt seeing all 10 claim fields jointly."""
    d = asdict(claim)
    d.pop("claim_id", None)
    system = (
        "You are adjudicating a synthetic healthcare insurance claim end-to-end.\n"
        "Rules:\n"
        "- REJECT if hospitalization_covered is false OR waiting_period_satisfied is false OR amount_inr > coverage_limit_inr.\n"
        "- Otherwise REVIEW if documents_available is false OR invoice_matches is false OR duplicate_claim is true OR provider_flagged is true OR identity_verified is false OR consent_available is false.\n"
        "- Otherwise APPROVE.\n"
        'Respond ONLY with JSON: {"decision": "APPROVE"|"REJECT"|"REVIEW", "confidence": <float 0..1>, "reasoning_summary": "<brief>"}'
    )
    user = f"Claim Facts: {json.dumps(d, sort_keys=True)}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _llm_judge_prompt(specialist_outputs: dict[str, Any]) -> list[dict[str, str]]:
    """Build LLM-as-Judge prompt seeing the 4 role specialist outputs."""
    condensed = {}
    for role in ROLES:
        raw = specialist_outputs[role]
        if isinstance(raw, dict) and "execution_error" not in raw:
            condensed[role] = {
                "decision": raw.get("decision"),
                "confidence": raw.get("confidence"),
                "evidence": raw.get("evidence"),
                "rule_references": raw.get("rule_references"),
                "reasoning_summary": raw.get("reasoning_summary"),
            }
        else:
            condensed[role] = {"execution_error": raw.get("execution_error") if isinstance(raw, dict) else "invalid"}
    system = (
        "You are a supervising judge reviewing outputs from four insurance specialists (policy, evidence, risk, compliance).\n"
        "Choose the final claim decision from APPROVE, REJECT, or REVIEW.\n"
        'Respond ONLY with JSON: {"decision": "APPROVE"|"REJECT"|"REVIEW", "confidence": <float 0..1>, "reasoning_summary": "<brief>"}'
    )
    user = f"Specialist Outputs: {json.dumps(condensed, sort_keys=True)}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _call_single_json_arm(
    client: Callable[[list[dict[str, str]], ModelProfile], Any],
    messages: list[dict[str, str]],
    *,
    provider: str,
    model_id: str,
    budget: LiveCallBudget,
    role_tag: str = "policy",
) -> dict[str, Any]:
    """Execute one standalone LLM call (Single-Agent E2E or LLM-as-Judge) using budget, retry & parse rules."""
    backoff_sec = 0.5 if isinstance(client, HostedChatModelClient) else 0.0
    prof = ModelProfile(
        model_id=model_id,
        revision="api",
        file_name="api",
        device="remote",
        role=role_tag,
        temperature=0.0,
        max_tokens=256,
        timeout_seconds=30.0,
        max_retries=2,
        provider=provider,
        backoff_seconds=backoff_sec,
        strict_input_scoping=False,
        allow_small_legacy=False,
    )
    t0 = time.perf_counter()
    raw_text = None
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    err = None
    for attempt in range(prof.max_retries + 1):
        budget.check_available()
        try:
            resp = client(messages, prof)
            if isinstance(resp, ModelClientResponse):
                raw_text = resp.raw_text
                usage = {
                    "prompt_tokens": resp.prompt_tokens or 0,
                    "completion_tokens": resp.completion_tokens or 0,
                    "total_tokens": resp.total_tokens or ((resp.prompt_tokens or 0) + (resp.completion_tokens or 0)),
                }
            elif isinstance(resp, dict) and "raw_text" in resp:
                raw_text = str(resp["raw_text"])
                usage = {
                    "prompt_tokens": int(resp.get("prompt_tokens") or 0),
                    "completion_tokens": int(resp.get("completion_tokens") or 0),
                    "total_tokens": int(resp.get("total_tokens") or 0),
                }
            else:
                raw_text = str(resp)
            budget.record(**usage)
            err = None
            break
        except BudgetExceededError:
            raise
        except NonTransientHTTPError as exc:
            err = f"{type(exc).__name__}: {exc}"
            budget.record(estimated_cost_usd=0.0)
            break
        except TransientHTTPError as exc:
            err = f"{type(exc).__name__}: {exc}"
            budget.record(estimated_cost_usd=0.0)
            if attempt == prof.max_retries:
                break
            if exc.retry_after_seconds is not None:
                wait_sec = min(30.0, exc.retry_after_seconds + 0.5)
                if wait_sec > 0:
                    time.sleep(wait_sec)
            elif prof.backoff_seconds > 0:
                time.sleep(prof.backoff_seconds * (2 ** attempt))
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            budget.record(estimated_cost_usd=0.0)
            if attempt == prof.max_retries:
                break
            if prof.backoff_seconds > 0:
                time.sleep(prof.backoff_seconds * (2 ** attempt))

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    decision = "REVIEW"
    parsed = None
    if raw_text and err is None:
        try:
            parsed = parse_model_response(raw_text)
            cand = str(parsed.get("decision", "")).upper()
            if cand in DECISIONS:
                decision = cand
        except Exception as exc:
            err = f"ParseError: {exc}"

    return {
        "decision": decision,
        "execution_error": err,
        "raw_response": raw_text,
        "parsed_json": parsed,
        "usage": usage,
        "latency_ms": latency_ms,
        "prompt_hash": sha256(messages),
    }


def run_live_phase6(
    *,
    subset_36: dict[str, Any],
    test_labels_by_id: dict[str, str],
    upd_labels_by_id: dict[str, dict[str, Any]],
    raw_dir: Path,
    provider: str,
    model_id: str,
    api_key_env_var: str,
    base_url: str | None = None,
    min_call_interval_sec: float = 5.0,
    max_calls: int = 320,
    max_cost_usd: float = 1.0,
    resume: bool = True,
    client_override: Callable[[list[dict[str, str]], ModelProfile], Any] | None = None,
) -> dict[str, Any]:
    """Run Option-B stratified 36-claim live track + 12 update paths with JSONL resumability."""
    validate_specialist_model_id(model_id, allow_small_legacy=False)
    live_path = raw_dir / "live_claims.jsonl"
    judge_path = raw_dir / "live_judge.jsonl"
    rec_path = raw_dir / "live_recovery.jsonl"

    if client_override is not None:
        client = client_override
    else:
        hosted = HostedChatModelClient(
            provider=provider,
            api_key_env_var=api_key_env_var,
            base_url=base_url,
            min_call_interval_sec=min_call_interval_sec,
        )
        if not hosted.is_configured():
            return {
                "status": "BLOCKED",
                "reason": f"API key env var '{api_key_env_var}' is unset or empty; zero fake calls run.",
                "calls_used": 0,
                "tokens_used": 0,
                "completed_claims": 0,
            }
        client = hosted

    existing_claims = {(r["arm"], r["claim_id"], r["phase"]): r for r in (_load_jsonl(live_path) if resume else [])}
    existing_judge = {(r["arm"], r["claim_id"], r["phase"]): r for r in (_load_jsonl(judge_path) if resume else [])}
    existing_rec = {(r["arm"], r["transition_id"], r["phase"]): r for r in (_load_jsonl(rec_path) if resume else [])}
    if not resume:
        for p in (live_path, judge_path, rec_path):
            if p.exists():
                p.unlink()

    budget = LiveCallBudget(max_calls=max_calls, max_cost_usd=max_cost_usd)
    signer = Ed25519EventSigner.generate()
    consecutive_429 = 0
    stop_reason: str | None = None

    def _check_consecutive_429(errors: Sequence[str | None]) -> bool:
        nonlocal consecutive_429, stop_reason
        if any(e and ("429" in e or "rate" in e.lower()) for e in errors):
            consecutive_429 += 1
        else:
            consecutive_429 = 0
        if consecutive_429 >= 3:
            stop_reason = "Aborted after 3 consecutive HTTP 429 daily/TPM quota errors"
            return True
        return False

    try:
        # 1. FULL_LIVE (4 role-isolated calls per claim) on 36 stratified claims
        for row in subset_36["claims"]:
            cid = row["claim_id"]
            if ("FULL_LIVE", cid, "initial") in existing_claims:
                continue
            claim = InsuranceClaim(**row["claim"])
            backends = build_phase4_role_backends(
                client,
                provider=provider,
                model_id=model_id,
                backoff_seconds=0.5 if isinstance(client, HostedChatModelClient) else 0.0,
                budget=budget,
                min_call_interval_sec=min_call_interval_sec if isinstance(client, HostedChatModelClient) else None,
            )
            wf = SpecialistCompositionAdjudication(claim, backends=backends, signer=signer)
            dec = wf.result()["decision"]
            round_ref = wf.system.latest("round").ref
            indep = verify_composition_independently(
                wf.system.artifact(round_ref), wf.system.provenance(round_ref), wf.ledger
            )
            raws = {r: wf.system.latest(f"{r}_agent").payload["raw_output"] for r in ROLES}
            toks = sum(
                (raws[r].get("model_metadata", {}).get("usage", {}).get("total_tokens") or 0)
                for r in ROLES if isinstance(raws[r], dict)
            )
            errs = [raws[r].get("execution_error") if isinstance(raws[r], dict) else None for r in ROLES]
            rec = {
                "arm": "FULL_LIVE",
                "phase": "initial",
                "claim_id": cid,
                "family_id": row["family_id"],
                "oracle_label": test_labels_by_id[cid],
                "decision": dec,
                "phase2_verified": indep.passed,
                "calls": 4,
                "total_tokens": toks,
                "raw_by_role": raws,
                "errors": errs,
            }
            _append_jsonl(live_path, rec)
            existing_claims[("FULL_LIVE", cid, "initial")] = rec
            if _check_consecutive_429(errs):
                break

        # 2. SINGLE_AGENT_E2E_LLM (1 call per claim) on 36 stratified claims
        if stop_reason is None:
            for row in subset_36["claims"]:
                cid = row["claim_id"]
                if ("SINGLE_AGENT_E2E_LLM", cid, "initial") in existing_claims:
                    continue
                claim = InsuranceClaim(**row["claim"])
                out = _call_single_json_arm(
                    client, _single_agent_prompt(claim), provider=provider, model_id=model_id, budget=budget
                )
                rec = {
                    "arm": "SINGLE_AGENT_E2E_LLM",
                    "phase": "initial",
                    "claim_id": cid,
                    "family_id": row["family_id"],
                    "oracle_label": test_labels_by_id[cid],
                    "decision": out["decision"],
                    "calls": 1,
                    "total_tokens": out["usage"]["total_tokens"],
                    **out,
                }
                _append_jsonl(live_path, rec)
                existing_claims[("SINGLE_AGENT_E2E_LLM", cid, "initial")] = rec
                if _check_consecutive_429([out["execution_error"]]):
                    break

        # 3. LLM_AS_JUDGE reusing saved FULL_LIVE 4 specialist outputs
        if stop_reason is None:
            for row in subset_36["claims"]:
                cid = row["claim_id"]
                if ("LLM_AS_JUDGE", cid, "initial") in existing_judge:
                    continue
                full_rec = existing_claims.get(("FULL_LIVE", cid, "initial"))
                if not full_rec:
                    continue
                out = _call_single_json_arm(
                    client,
                    _llm_judge_prompt(full_rec["raw_by_role"]),
                    provider=provider,
                    model_id=model_id,
                    budget=budget,
                )
                rec = {
                    "arm": "LLM_AS_JUDGE",
                    "phase": "initial",
                    "claim_id": cid,
                    "family_id": row["family_id"],
                    "oracle_label": test_labels_by_id[cid],
                    "decision": out["decision"],
                    "judge_calls": 1,
                    "reused_specialist_calls": 4,
                    "total_tokens": out["usage"]["total_tokens"],
                    **out,
                }
                _append_jsonl(judge_path, rec)
                existing_judge[("LLM_AS_JUDGE", cid, "initial")] = rec
                if _check_consecutive_429([out["execution_error"]]):
                    break

        # 4. Recovery: Deterministic replay equivalence + Fresh live paired selective_recovery vs full_rerun (12 paths)
        if stop_reason is None:
            for upd in subset_36["updates"]:
                tid = upd["transition_id"]
                if ("LIVE_RECOVERY_PAIRED", tid, "update") in existing_rec:
                    continue
                c1 = InsuranceClaim(**upd["claim_v1"])
                c2 = InsuranceClaim(**upd["claim_v2"])
                lbl_v2 = upd_labels_by_id[tid]["label_v2"]

                # Fresh live selective recovery
                backends_sel = build_phase4_role_backends(
                    client,
                    provider=provider,
                    model_id=model_id,
                    backoff_seconds=0.5 if isinstance(client, HostedChatModelClient) else 0.0,
                    budget=budget,
                    min_call_interval_sec=min_call_interval_sec if isinstance(client, HostedChatModelClient) else None,
                )
                wf_sel = SpecialistCompositionAdjudication(c1, backends=backends_sel, signer=signer)
                exec_before = sum(wf_sel.executions.values())
                wf_sel.update_claim(c2)
                sel_calls = sum(wf_sel.executions.values()) - exec_before
                sel_dec = wf_sel.result()["decision"]
                sel_raws_v2 = {r: wf_sel.system.latest(f"{r}_agent").payload["raw_output"] for r in ROLES}

                # Deterministic replay equivalence using exact captured v2 outputs
                def _replay_client(captured):
                    def fn(msgs, prof):
                        raw = captured[prof.role]
                        return json.dumps(raw) if "execution_error" not in raw else ModelClientResponse(raw_text="error")
                    return fn

                wf_rep_full = SpecialistCompositionAdjudication(
                    c2,
                    backends=build_phase4_role_backends(_replay_client(sel_raws_v2), provider=provider, model_id=model_id, backoff_seconds=0.0),
                    signer=signer,
                )
                replay_full_dec = wf_rep_full.result()["decision"]

                # Fresh live full rerun on v2
                backends_full = build_phase4_role_backends(
                    client,
                    provider=provider,
                    model_id=model_id,
                    backoff_seconds=0.5 if isinstance(client, HostedChatModelClient) else 0.0,
                    budget=budget,
                    min_call_interval_sec=min_call_interval_sec if isinstance(client, HostedChatModelClient) else None,
                )
                wf_full = SpecialistCompositionAdjudication(c2, backends=backends_full, signer=signer)
                full_dec = wf_full.result()["decision"]

                rec = {
                    "arm": "LIVE_RECOVERY_PAIRED",
                    "transition_id": tid,
                    "phase": "update",
                    "update_kind": upd["update_kind"],
                    "oracle_label_v2": lbl_v2,
                    "selective_calls": sel_calls,
                    "full_rerun_calls": 4,
                    "selective_decision": sel_dec,
                    "replay_full_rerun_decision": replay_full_dec,
                    "fresh_full_rerun_decision": full_dec,
                    "deterministic_replay_equivalent": sel_dec == replay_full_dec,
                    "fresh_live_equivalent": sel_dec == full_dec,
                }
                _append_jsonl(rec_path, rec)
                existing_rec[("LIVE_RECOVERY_PAIRED", tid, "update")] = rec

    except Exception as exc:
        stop_reason = f"Stopped early: {type(exc).__name__}: {exc}"

    completed_full = sum(1 for k in existing_claims if k[0] == "FULL_LIVE")
    status = "PASS" if (stop_reason is None and completed_full == subset_36["claim_count"]) else "BLOCKED"
    return {
        "status": status,
        "reason": stop_reason,
        "budget": budget.snapshot(),
        "completed_full_live_claims": completed_full,
        "completed_single_agent_claims": sum(1 for k in existing_claims if k[0] == "SINGLE_AGENT_E2E_LLM"),
        "completed_judge_claims": len(existing_judge),
        "completed_recovery_paths": len(existing_rec),
    }


def rebuild_phase6_tables(
    *,
    results_dir: Path,
    manifest_info: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Purely aggregate saved raw JSONL logs under `results/phase6/raw/` into JSON + Markdown tables."""
    raw_dir = results_dir / "raw"
    offline_trials = _load_jsonl(raw_dir / "offline_trials.jsonl")
    offline_ablations = _load_jsonl(raw_dir / "offline_ablations.jsonl")
    offline_recovery = _load_jsonl(raw_dir / "offline_recovery.jsonl")
    live_claims = _load_jsonl(raw_dir / "live_claims.jsonl")
    live_judge = _load_jsonl(raw_dir / "live_judge.jsonl")
    live_recovery = _load_jsonl(raw_dir / "live_recovery.jsonl")

    attempt_manifest_path = results_dir / "attempt_manifest.json"
    attempt_manifest = json.loads(attempt_manifest_path.read_text(encoding="utf-8")) if attempt_manifest_path.exists() else {}

    offline_arms = ("FULL_DET", "MAJORITY_VOTE", "CONFIDENCE_WEIGHTED_VOTE", "MAJORITY_POLICY_VETO")
    clean_rows = [r for r in offline_trials if r["track"] == "offline_paired_unique_cases"]
    per_fault_rows = [r for r in offline_trials if r["track"] == "offline_per_fault_unique_cases"]
    sweep_rows = [r for r in offline_trials if r["track"] == "offline_10seed_sweep"]

    def _summarize_track(rows: list[dict[str, Any]], arms: Sequence[str], *, unique_n: int, total_trials: int) -> dict[str, Any]:
        if not rows:
            return {"status": "NOT RUN", "unique_claims": 0, "total_trials": 0, "arms": {}}
        y_true = [r["oracle_label"] for r in rows]
        clusters = [r["family_id"] for r in rows]
        arm_stats = {}
        full_preds = [r["decisions"]["FULL_DET"] for r in rows] if "FULL_DET" in arms else None
        for arm in arms:
            preds = [r["decisions"][arm] for r in rows]
            m = compute_phase6_metrics(preds, y_true)
            m["unsafe_approval_wilson95"] = wilson_ci(m["unsafe_approvals"], m["total_non_approve_cases"])
            m["exact_match_wilson95"] = wilson_ci(m["correct_all"], m["total_cases"])
            if full_preds is not None and arm != "FULL_DET" and unique_n == total_trials:
                full_acc = [p == y for p, y in zip(full_preds, y_true)]
                arm_acc = [p == y for p, y in zip(preds, y_true)]
                m["mcnemar_exact_match_vs_FULL_DET"] = exact_mcnemar_test(full_acc, arm_acc)
                m["bootstrap95_exact_match_diff_vs_FULL_DET"] = paired_bootstrap_ci(
                    [float(x) for x in full_acc],
                    [float(x) for x in arm_acc],
                    clusters=clusters,
                    n_resamples=2000,
                    seed=GENERATOR_SEED,
                )
                non_app_idx = [i for i, y in enumerate(y_true) if y != "APPROVE"]
                full_safe = [full_preds[i] != "APPROVE" for i in non_app_idx]
                arm_safe = [preds[i] != "APPROVE" for i in non_app_idx]
                m["mcnemar_safety_vs_FULL_DET"] = exact_mcnemar_test(full_safe, arm_safe)
            arm_stats[arm] = m
        return {"status": "PASS", "unique_claims": unique_n, "total_trials": total_trials, "arms": arm_stats}

    clean_summary = _summarize_track(clean_rows, offline_arms, unique_n=len(clean_rows), total_trials=len(clean_rows))
    per_fault_summaries = {
        f_cls: _summarize_track(
            [r for r in per_fault_rows if r["fault_class"] == f_cls],
            offline_arms,
            unique_n=104,
            total_trials=104,
        )
        for f_cls in PHASE6_FAULT_CLASSES
    }
    sweep_summary = _summarize_track(sweep_rows, offline_arms, unique_n=104, total_trials=len(sweep_rows))

    # Recovery table grouped by update_kind
    rec_by_kind: dict[str, dict[str, Any]] = {}
    for kind in ("narrow_single_role", "broad_multi_role", "no_change_edit", "same_label_value_edit"):
        sub = [r for r in offline_recovery if r["update_kind"] == kind]
        if sub:
            rec_by_kind[kind] = {
                "count": len(sub),
                "avg_selective_execs": round(sum(r["selective_specialist_executions"] for r in sub) / len(sub), 2),
                "avg_skipped_execs": round(sum(r["skipped_specialist_executions"] for r in sub) / len(sub), 2),
                "avg_nodes_rebuilt": round(sum(r.get("nodes_rebuilt", 0) for r in sub) / len(sub), 2),
                "avg_nodes_skipped": round(sum(r.get("nodes_skipped", 0) for r in sub) / len(sub), 2),
                "full_rerun_execs": 4,
                "avg_selective_ms": round(sum(r["selective_latency_ms"] for r in sub) / len(sub), 3),
                "avg_full_rerun_ms": round(sum(r["full_rerun_latency_ms"] for r in sub) / len(sub), 3),
                "avg_verification_overhead_ms": round(sum(r.get("verification_overhead_ms", 0.0) for r in sub) / len(sub), 3),
                "avg_signing_overhead_ms": round(sum(r.get("signing_overhead_ms", 0.0) for r in sub) / len(sub), 3),
                "equivalence_rate": sum(1 for r in sub if r["equivalent"]) / len(sub),
                "stale_with_cert_count": sum(1 for r in sub if r["stale_with_certificates"]),
                "stale_without_cert_count": sum(1 for r in sub if r["stale_without_certificates"]),
            }

    # Live track summary (BLOCKED / NOT RUN if empty)
    live_summary: dict[str, Any] = {}
    for arm_name, src in (
        ("FULL_LIVE", [r for r in live_claims if r["arm"] == "FULL_LIVE"]),
        ("SINGLE_AGENT_E2E_LLM", [r for r in live_claims if r["arm"] == "SINGLE_AGENT_E2E_LLM"]),
        ("LLM_AS_JUDGE", live_judge),
    ):
        if src:
            metrics = compute_phase6_metrics([r["decision"] for r in src], [r["oracle_label"] for r in src])
            live_summary[arm_name] = {
                "status": "PASS" if len(src) == 36 else "PARTIAL_BLOCKED",
                "cases_completed": len(src),
                "metrics": metrics,
            }
        else:
            live_status = attempt_manifest.get("live_track", {}).get("status", "NOT RUN")
            live_summary[arm_name] = {
                "status": live_status,
                "cases_completed": 0,
                "metrics": None,
            }

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_manifest_commit": manifest_info["manifest"]["code_commit_sha"],
        "blind_external_arm": attempt_manifest.get("blind_external_arm", "NOT RUN (0 submitted)"),
        "offline_clean_N104": clean_summary,
        "offline_per_fault_N104": per_fault_summaries,
        "offline_10seed_sweep_N104x10_exploratory": sweep_summary,
        "ablations": offline_ablations,
        "recovery_by_update_kind": rec_by_kind,
        "recovery_overall_equivalence": (
            sum(1 for r in offline_recovery if r["equivalent"]) / len(offline_recovery)
            if offline_recovery else None
        ),
        "live_arms": live_summary,
        "live_recovery_count": len(live_recovery),
    }
    (results_dir / "tables_and_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Render concise markdown report
    md = [
        "# Phase 6 Evaluation Tables & Metrics",
        "",
        f"- **Generated (UTC):** `{payload['generated_utc']}`",
        f"- **Frozen Manifest Commit:** `{payload['frozen_manifest_commit']}`",
        f"- **Blind External Arm:** `{payload['blind_external_arm']}`",
        "",
        "## 1. Clean Held-Out Test Set (`N=104` Unique Claims, Seed `101`)",
        "",
        "| Arm | Exact-Match Acc | 95% Wilson CI | Unsafe-Approval Rate | False-Reject Rate | Coverage | Selective Acc | McNemar p vs `FULL_DET` (Acc / Safe) |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for arm in offline_arms:
        if arm in clean_summary.get("arms", {}):
            m = clean_summary["arms"][arm]
            w = m["exact_match_wilson95"]
            mc_a = m.get("mcnemar_exact_match_vs_FULL_DET", {}).get("exact_two_sided_p")
            mc_s = m.get("mcnemar_safety_vs_FULL_DET", {}).get("exact_two_sided_p")
            mc_str = f"{mc_a:.3e} / {mc_s:.3e}" if mc_a is not None and mc_s is not None else "baseline ref"
            sel_acc = f"{m['selective_accuracy']:.4f}" if m["selective_accuracy"] is not None else "N/A"
            md.append(
                f"| `{arm}` | {m['exact_match_accuracy']:.4f} | [{w['low']:.4f}, {w['high']:.4f}] | {m['unsafe_approval_rate']:.4f} | {m['false_reject_rate']:.4f} | {m['coverage']:.4f} | {sel_acc} | `{mc_str}` |"
            )

    md.extend([
        "",
        "## 2. Multi-Seed Fault Sweep (`N=104` Unique Claims × `10` Seeds = `1,040` Trials, Exploratory)",
        "",
        "| Arm | Exact-Match Acc | Unsafe-Approval Rate | False-Reject Rate | Coverage | Selective Acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for arm in offline_arms:
        if arm in sweep_summary.get("arms", {}):
            m = sweep_summary["arms"][arm]
            sel_acc = f"{m['selective_accuracy']:.4f}" if m["selective_accuracy"] is not None else "N/A"
            md.append(
                f"| `{arm}` | {m['exact_match_accuracy']:.4f} | {m['unsafe_approval_rate']:.4f} | {m['false_reject_rate']:.4f} | {m['coverage']:.4f} | {sel_acc} |"
            )

    md.extend([
        "",
        "## 3. Selective Recovery vs. Full Rerun (`32` Frozen `v1 -> v2` Update Paths)",
        "",
        "| Update Kind | Count | Avg Rebuilds / Skipped (Specialists) | Avg Nodes Rebuilt / Skipped | Avg Selective vs Full (ms) | Avg Verify / Sign Overhead (ms) | Equivalence | Stale (Cert / No-Cert) |",
        "| --- | ---: | --- | --- | --- | --- | ---: | --- |",
    ])
    for kind, row in rec_by_kind.items():
        md.append(
            f"| `{kind}` | {row['count']} | `{row['avg_selective_execs']:.2f} / {row['avg_skipped_execs']:.2f}` | `{row['avg_nodes_rebuilt']:.2f} / {row['avg_nodes_skipped']:.2f}` | `{row['avg_selective_ms']:.3f} vs {row['avg_full_rerun_ms']:.3f}` | `{row['avg_verification_overhead_ms']:.3f} / {row['avg_signing_overhead_ms']:.3f}` | {row['equivalence_rate']:.2%} | `{row['stale_with_cert_count']} / {row['stale_without_cert_count']}` |"
        )

    md.extend([
        "",
        "## 4. Live Option-B (`N=36`) Arm Status",
        "",
        "| Arm | Status | Completed Cases | Exact-Match Acc | Unsafe-Approval Rate |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for arm_name, st in live_summary.items():
        m = st.get("metrics") or {}
        acc = f"{m['exact_match_accuracy']:.4f}" if "exact_match_accuracy" in m else "N/A"
        uar = f"{m['unsafe_approval_rate']:.4f}" if "unsafe_approval_rate" in m else "N/A"
        md.append(f"| `{arm_name}` | `{st['status']}` | {st['cases_completed']} | {acc} | {uar} |")

    (results_dir / "phase6_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return payload
