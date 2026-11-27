"""Frozen Phase 5/6 evaluation protocol: metrics, statistical tests, baselines, and Groq budget.

Clean offline evaluation/oracle module inside `cert_recovery.oracle`.
"""
from __future__ import annotations

from collections import defaultdict
import math
import random
from typing import Any, Callable, Sequence

from .frozen_v1 import SWEEP_SEEDS

PROTOCOL_VERSION = "protocol-v1"
BOOTSTRAP_RESAMPLES = 2000
WILSON_Z_95 = 1.959963984540054


PRIMARY_BASELINES_V1: tuple[dict[str, Any], ...] = (
    {
        "arm_id": "FULL_DET",
        "name": "Full System (Deterministic Rule Specialists)",
        "category": "primary_system",
        "input_visibility": "strict_role_scoped",
        "decision_authority": "composition-v1 (substantive_exclusion)",
        "validation_rules": "assess_specialist + verify_composition_independently",
        "llm_calls_per_claim": 0,
    },
    {
        "arm_id": "FULL_LIVE",
        "name": "Full System (Role-Isolated groq:openai/gpt-oss-20b)",
        "category": "primary_system",
        "input_visibility": "strict_role_scoped",
        "decision_authority": "composition-v1 (substantive_exclusion)",
        "validation_rules": "assess_specialist + verify_composition_independently + Ed25519 EventDAG",
        "llm_calls_per_claim": 4,
    },
    {
        "arm_id": "MAJORITY_VOTE",
        "name": "Majority Vote (No Exclusion Gate)",
        "category": "baseline",
        "input_visibility": "strict_role_scoped",
        "decision_authority": "unweighted_majority (ties -> REVIEW)",
        "validation_rules": "none_on_gate",
        "llm_calls_per_claim": "0 (reuses role outputs) or 4",
    },
    {
        "arm_id": "CONFIDENCE_WEIGHTED_VOTE",
        "name": "Confidence-Weighted Vote (No Exclusion Gate)",
        "category": "baseline",
        "input_visibility": "strict_role_scoped",
        "decision_authority": "confidence_weighted_consensus (no mandatory gate)",
        "validation_rules": "confidence weighting only",
        "llm_calls_per_claim": "0 (reuses role outputs) or 4",
    },
    {
        "arm_id": "MAJORITY_POLICY_VETO",
        "name": "Majority with Validated Policy Veto",
        "category": "strong_non_strawman_baseline",
        "input_visibility": "strict_role_scoped",
        "decision_authority": "validated policy REJECT veto + unweighted majority otherwise",
        "validation_rules": "policy obligation validated against coverage-v1; other votes majority",
        "llm_calls_per_claim": "0 (reuses role outputs) or 4",
    },
    {
        "arm_id": "SINGLE_AGENT_E2E_LLM",
        "name": "Single-Agent End-to-End LLM",
        "category": "baseline_llm",
        "input_visibility": "full_unscoped_claim (all 10 fields)",
        "decision_authority": "single monolithic LLM decision",
        "validation_rules": "JSON schema parse only",
        "llm_calls_per_claim": 1,
    },
    {
        "arm_id": "LLM_AS_JUDGE",
        "name": "LLM-as-Judge over 4 Specialist Outputs",
        "category": "baseline_llm_budget_capped",
        "input_visibility": "4 specialist outputs -> 1 judge LLM prompt",
        "decision_authority": "judge LLM verdict (no deterministic gate)",
        "validation_rules": "JSON schema parse on judge (+1 call reusing 4 specialist outputs)",
        "llm_calls_per_claim": "1 judge (+4 specialist = 5 total, or 1 if reusing FULL_LIVE outputs)",
    },
)

DEPRECATED_HISTORICAL_ARM_NOTE = {
    "arm_id": "LEGACY_ARM_E_EXCLUSION_BLIND",
    "status": "DROPPED_FROM_PRIMARY_COMPARISONS",
    "reason": "Exclusion-blind whole-claim weighted consensus (`mode='legacy'` from v0-paper) is retained strictly as a deprecated historical reference note.",
}

TARGETED_ABLATIONS_V1: tuple[dict[str, str], ...] = (
    {
        "ablation_id": "no_gate",
        "delta_from_full": "Bypass composition-v1 mandatory policy-exclusion gate (fall back to advisory consensus).",
        "primary_metric_isolated": "unsafe_approval_rate",
    },
    {
        "ablation_id": "no_validation",
        "delta_from_full": "Skip assess_specialist schema/rule/binding checks on specialist JSON.",
        "primary_metric_isolated": "malformed_and_invalid_containment_rate",
    },
    {
        "ablation_id": "no_certificates",
        "delta_from_full": "Disable CertificateSystem dependency hash & version staleness checks.",
        "primary_metric_isolated": "stale_decision_rate",
    },
    {
        "ablation_id": "no_provenance",
        "delta_from_full": "Disable Ed25519 EventDAG signatures & persisted checkpointing.",
        "primary_metric_isolated": "tamper_detection_and_lineage_auditability",
    },
    {
        "ablation_id": "no_recovery",
        "delta_from_full": "Execute full clean rerun instead of selective dependency recovery on claim updates.",
        "primary_metric_isolated": "post_update_calls_tokens_latency_and_equivalence",
    },
)


def compute_phase6_metrics(predictions: Sequence[str], oracle_labels: Sequence[str]) -> dict[str, Any]:
    """Compute frozen Phase 6 metrics with exact conditioned and raw denominators."""
    if len(predictions) != len(oracle_labels):
        raise ValueError("predictions and oracle_labels must have equal length")
    n = len(predictions)
    if n == 0:
        raise ValueError("Cannot compute metrics over 0 cases")

    non_approve_total = sum(1 for y in oracle_labels if y != "APPROVE")
    non_reject_total = sum(1 for y in oracle_labels if y != "REJECT")

    unsafe_approvals = sum(1 for p, y in zip(predictions, oracle_labels) if p == "APPROVE" and y != "APPROVE")
    false_rejects = sum(1 for p, y in zip(predictions, oracle_labels) if p == "REJECT" and y != "REJECT")

    decided_pairs = [(p, y) for p, y in zip(predictions, oracle_labels) if p in ("APPROVE", "REJECT")]
    decided_count = len(decided_pairs)
    reviewed_count = sum(1 for p in predictions if p == "REVIEW")
    correct_decided = sum(1 for p, y in decided_pairs if p == y)
    correct_all = sum(1 for p, y in zip(predictions, oracle_labels) if p == y)

    return {
        "total_cases": n,
        "total_non_approve_cases": non_approve_total,
        "total_non_reject_cases": non_reject_total,
        "unsafe_approvals": unsafe_approvals,
        "unsafe_approval_rate": (unsafe_approvals / non_approve_total) if non_approve_total else 0.0,
        "unsafe_approval_rate_over_total": unsafe_approvals / n,
        "false_rejects": false_rejects,
        "false_reject_rate": (false_rejects / non_reject_total) if non_reject_total else 0.0,
        "decided_count": decided_count,
        "coverage": decided_count / n,
        "reviewed_count": reviewed_count,
        "review_rate": reviewed_count / n,
        "correct_decided": correct_decided,
        "selective_accuracy": (correct_decided / decided_count) if decided_count else None,
        "correct_all": correct_all,
        "exact_match_accuracy": correct_all / n,
    }


def wilson_ci(successes: int, total: int, z: float = WILSON_Z_95) -> dict[str, Any]:
    """Compute exact 95% Wilson score interval for a binomial proportion."""
    if total <= 0:
        return {"estimate": None, "low": None, "high": None, "k": successes, "n": total}
    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / total) + (z2 / (4.0 * total * total)))
    return {
        "estimate": p,
        "low": 0.0 if successes == 0 else max(0.0, center - margin),
        "high": 1.0 if successes == total else min(1.0, center + margin),
        "k": successes,
        "n": total,
    }


def paired_bootstrap_ci(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    clusters: Sequence[str] | None = None,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 20261127,
) -> dict[str, Any]:
    """Compute 95% paired percentile bootstrap CI for mean(values_a - values_b).

    If `clusters` is provided (e.g., template family IDs), performs family-clustered
    block resampling across clusters cleanly.
    """
    if len(values_a) != len(values_b) or not values_a:
        raise ValueError("values_a and values_b must be non-empty and equal length")
    diffs = [float(a) - float(b) for a, b in zip(values_a, values_b)]
    observed = sum(diffs) / len(diffs)
    rng = random.Random(seed)

    if clusters is not None:
        if len(clusters) != len(diffs):
            raise ValueError("clusters length must match values length")
        by_cluster: dict[str, list[float]] = defaultdict(list)
        for cl, d in zip(clusters, diffs):
            by_cluster[cl].append(d)
        cluster_lists = list(by_cluster.values())
        k = len(cluster_lists)
        boot_means: list[float] = []
        for _ in range(n_resamples):
            sampled = []
            for _ in range(k):
                sampled.extend(rng.choice(cluster_lists))
            boot_means.append(sum(sampled) / len(sampled))
    else:
        n = len(diffs)
        boot_means = [
            sum(rng.choice(diffs) for _ in range(n)) / n
            for _ in range(n_resamples)
        ]

    boot_means.sort()
    lo_idx = max(0, int(0.025 * (n_resamples - 1)))
    hi_idx = min(n_resamples - 1, int(0.975 * (n_resamples - 1)))
    return {
        "observed_diff": observed,
        "ci95_low": boot_means[lo_idx],
        "ci95_high": boot_means[hi_idx],
        "n_items": len(diffs),
        "clustered": clusters is not None,
        "n_resamples": n_resamples,
    }


def exact_mcnemar_test(outcomes_a: Sequence[bool], outcomes_b: Sequence[bool]) -> dict[str, Any]:
    """Exact two-sided McNemar test over one paired binary outcome per independent case."""
    if len(outcomes_a) != len(outcomes_b):
        raise ValueError("outcomes_a and outcomes_b must have equal length")
    n10 = sum(1 for a, b in zip(outcomes_a, outcomes_b) if bool(a) and not bool(b))
    n01 = sum(1 for a, b in zip(outcomes_a, outcomes_b) if not bool(a) and bool(b))
    n_disc = n10 + n01
    if n_disc == 0:
        p_value = 1.0
    else:
        m = min(n10, n01)
        tail = sum(math.comb(n_disc, k) * (0.5 ** n_disc) for k in range(0, m + 1))
        p_value = min(1.0, 2.0 * tail)
    return {
        "n_total": len(outcomes_a),
        "n_a_only": n10,
        "n_b_only": n01,
        "discordant_pairs": n_disc,
        "exact_two_sided_p": p_value,
    }


def build_groq_8k_tpm_budget_plan(
    *,
    test_count: int = 104,
    update_count: int = 32,
    stratified_subset_count: int = 36,
    avg_tokens_per_call: float = 550.0,
    min_call_interval_sec: float = 5.0,
    tpm_limit: int = 8000,
    update_avg_calls_selective: float = 1.8,
) -> dict[str, Any]:
    """Build exact Groq free-tier 8k-TPM budget calibrated to Phase 4 telemetry."""
    calls_per_min = 60.0 / min_call_interval_sec
    effective_tpm = calls_per_min * avg_tokens_per_call

    # Option A: Full 104-claim primary live run + selective recovery vs no_recovery on update slice
    opt_a_first_run_calls = test_count * 4
    opt_a_selective_update_calls = int(round(update_count * update_avg_calls_selective))
    opt_a_total_calls_primary = opt_a_first_run_calls + opt_a_selective_update_calls
    opt_a_all_3_live_arms_calls = (test_count * 4) + test_count + test_count + opt_a_selective_update_calls

    # Option B: Stratified 36-claim live subset across all 3 live LLM arms + selective updates
    opt_b_full_live_calls = stratified_subset_count * 4
    opt_b_single_agent_calls = stratified_subset_count
    opt_b_judge_calls = stratified_subset_count  # reusing FULL_LIVE specialist outputs
    opt_b_update_calls = 24  # 12 update paths (~1.8 calls average rounded up)
    opt_b_total_calls = opt_b_full_live_calls + opt_b_single_agent_calls + opt_b_judge_calls + opt_b_update_calls

    def _entry(calls: int) -> dict[str, Any]:
        tokens = int(round(calls * avg_tokens_per_call))
        minutes = round((calls * min_call_interval_sec) / 60.0, 1)
        return {
            "calls": calls,
            "estimated_tokens": tokens,
            "estimated_cost_usd": 0.0,
            "estimated_wall_clock_minutes": minutes,
        }

    return {
        "authorized_provider": "groq",
        "authorized_model_id": "openai/gpt-oss-20b",
        "calibration_source": "results/phase4/pilot_summary.json & live_pilot_calls.jsonl (525.54 dev avg tokens/call, 4.73s avg latency)",
        "planning_assumptions": {
            "avg_tokens_per_call": avg_tokens_per_call,
            "min_call_interval_sec": min_call_interval_sec,
            "calls_per_minute": calls_per_min,
            "projected_tpm": round(effective_tpm, 1),
            "groq_free_tier_tpm_limit": tpm_limit,
            "safety_margin_below_tpm_limit": round(tpm_limit - effective_tpm, 1),
        },
        "option_a_full_test_104_claims": {
            "FULL_LIVE_first_run_only": _entry(opt_a_first_run_calls),
            "FULL_LIVE_plus_selective_updates": _entry(opt_a_total_calls_primary),
            "all_3_live_arms_plus_updates": _entry(opt_a_all_3_live_arms_calls),
            "note": "Running all 3 live LLM arms on 104 claims (~674 calls / ~371k tokens / ~56 min) risks Groq free-tier daily RPD/TPD exhaustion; prioritize FULL_LIVE on 104 claims OR invoke Option B if daily quota triggers.",
        },
        "option_b_stratified_36_claim_live_subset": {
            "FULL_LIVE": _entry(opt_b_full_live_calls),
            "SINGLE_AGENT_E2E_LLM": _entry(opt_b_single_agent_calls),
            "LLM_AS_JUDGE_reusing_specialists": _entry(opt_b_judge_calls),
            "selective_recovery_updates": _entry(opt_b_update_calls),
            "total_all_live_arms": _entry(opt_b_total_calls),
            "note": "Balanced across all 8 held-out test families (~20 min, 240 calls, ~132k tokens) safely within single-session Groq free-tier TPM/RPD.",
        },
        "stopping_rules": [
            "Enforce LiveCallBudget hard caps per run (max_calls and max_cost_usd) before each outbound HTTP call.",
            "On 3 consecutive HTTP 429 daily/TPD quota errors (after honoring server Retry-After hint <= 30s), halt remaining live calls cleanly and record partial live coverage without fabricating missing results.",
            "NEVER execute live inference or frozen test-set runs inside Phase 5; reserve execution strictly for Phase 6.",
        ],
    }


def frozen_statistical_protocol() -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "sweep_seeds": list(SWEEP_SEEDS),
        "num_sweep_seeds": len(SWEEP_SEEDS),
        "binomial_ci_method": "95% Wilson score interval",
        "bootstrap_ci_method": "Paired percentile bootstrap (B=2000, supporting family-clustered resampling)",
        "hypothesis_test": "Exact two-sided McNemar test on paired per-case binary outcomes (exploratory tag required for multi-seed/subgroup views)",
    }
