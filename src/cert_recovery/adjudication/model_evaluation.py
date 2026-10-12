"""Bounded evaluation harness comparing deterministic vs model-backed role agent."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
from typing import Callable

from .ablation import extended_scenarios, historical_cases
from .consensus import assess
from .domain import InsuranceClaim, ROLES, RuleAgent, rule_decision
from .model_agent import ModelAgentAdapter, ModelProfile
from .workflow import InsuranceAdjudication


def compliance_evaluation_cases():
    """Frozen set of 11 claims exercising all compliance states and whole-claim conditions."""
    base = InsuranceClaim()
    cases = []
    # 10 extended scenarios
    for item in extended_scenarios():
        cases.append({
            "case_id": item["case_id"],
            "claim": item["claim"],
            "description": item["description"],
            "expected_role_decision": rule_decision("compliance", item["claim"].facts("compliance")),
            "expected_claim_label": item["label"],
        })
    # Plus historical missing-ID case
    cases.append({
        "case_id": "hist_4_missing_id",
        "claim": replace(base, identity_verified=False),
        "description": "Historical compliance unverified identity",
        "expected_role_decision": "REVIEW",
        "expected_claim_label": "REVIEW",
    })
    return cases


def mock_model_client(messages: list[dict[str, str]], profile: ModelProfile) -> str:
    """Deterministic LLM response simulator adhering to the Qwen-2.5 instruction contract."""
    user_content = messages[-1]["content"]
    if "Facts to Evaluate: " in user_content:
        facts = json.loads(user_content.split("Facts to Evaluate: ")[1].split("\n\n")[0])
    elif "Facts: " in user_content:
        facts = json.loads(user_content.split("Facts: ")[1].strip())
    else:
        facts = {}
    time.sleep(0.005)  # Simulate 5ms inference latency

    identity = facts.get("identity_verified", False)
    consent = facts.get("consent_available", False)

    if identity and consent:
        decision = "APPROVE"
        confidence = 0.88  # Realistic self-reported model confidence
        reasoning = "Both identity verification and patient consent are documented."
    else:
        decision = "REVIEW"
        confidence = 0.72  # Self-reported uncertainty on review trigger
        reasoning = f"Missing required compliance controls: identity={identity}, consent={consent}."

    return json.dumps({
        "decision": decision,
        "confidence": confidence,
        "evidence": facts,
        "rule_references": ["identity-consent-v1"],
        "reasoning_summary": reasoning,
    }, indent=2)


def run_model_comparison(*, custom_client: Callable | None = None, live_backend: bool = False) -> dict:
    """Execute head-to-head evaluation between RuleAgent and ModelAgentAdapter."""
    cases = compliance_evaluation_cases()
    client = custom_client if custom_client is not None else (None if live_backend else mock_model_client)
    adapter = ModelAgentAdapter(client=client)
    rule_agent = RuleAgent()

    results = []
    for c in cases:
        claim = c["claim"]
        cid = c["case_id"]
        facts = claim.facts("compliance")
        binding = {"artifact_id": "facts:compliance", "version": 1, "content_hash": "mock_hash"}
        expected_role = c["expected_role_decision"]

        # 1. Deterministic Rule Agent evaluation
        t0 = time.perf_counter()
        rule_out = rule_agent.execute("compliance", "Evaluate compliance", facts, binding)
        rule_ms = round((time.perf_counter() - t0) * 1000, 3)
        rule_assess = assess(rule_out, "compliance", facts, binding)

        # 2. Model Agent evaluation
        t0 = time.perf_counter()
        model_out = adapter.execute("compliance", "Evaluate compliance", facts, binding)
        model_ms = round((time.perf_counter() - t0) * 1000, 3)
        model_assess = assess(model_out, "compliance", facts, binding)

        # 3. Whole-pipeline integration comparison
        wf_rule = InsuranceAdjudication(claim)
        final_rule = wf_rule.result()

        wf_model = InsuranceAdjudication(claim, backends={"compliance": adapter})
        final_model = wf_model.result()

        results.append({
            "case_id": cid,
            "description": c["description"],
            "expected_role_decision": expected_role,
            "expected_claim_label": c["expected_claim_label"],
            "rule_agent": {
                "decision": rule_out["decision"],
                "confidence": rule_out["confidence"],
                "confidence_origin": rule_out["uncertainty"]["calibration"],
                "trust_status": rule_assess.trust_status,
                "validation_score": rule_assess.validation_score,
                "latency_ms": rule_ms,
                "whole_claim_decision": final_rule["decision"],
                "gate_status": final_rule["gate"],
            },
            "model_agent": {
                "decision": model_out.get("decision", "ERROR"),
                "confidence": model_out.get("confidence", 0.0),
                "confidence_origin": model_out.get("uncertainty", {}).get("confidence_origin", "unavailable"),
                "trust_status": model_assess.trust_status,
                "validation_score": model_assess.validation_score,
                "fault_score": model_assess.fault_score,
                "latency_ms": model_ms,
                "call_status": model_out.get("model_metadata", {}).get("call_status", "UNKNOWN"),
                "whole_claim_decision": final_model["decision"],
                "gate_status": final_model["gate"],
            },
            "comparison": {
                "role_agreement": rule_out["decision"] == model_out.get("decision"),
                "claim_agreement": final_rule["decision"] == final_model["decision"],
                "rule_correct": rule_out["decision"] == expected_role,
                "model_correct": model_out.get("decision") == expected_role,
            },
        })

    # Summary metrics
    n = len(results)
    model_correct = sum(r["comparison"]["model_correct"] for r in results)
    rule_correct = sum(r["comparison"]["rule_correct"] for r in results)
    agreements = sum(r["comparison"]["role_agreement"] for r in results)
    avg_model_lat = sum(r["model_agent"]["latency_ms"] for r in results) / n
    avg_rule_lat = sum(r["rule_agent"]["latency_ms"] for r in results) / n

    return {
        "evaluation_version": "model-eval-v1",
        "target_role": "compliance",
        "model_id": adapter.profile.model_id,
        "revision": adapter.profile.revision,
        "live_backend_executed": live_backend and client is not None,
        "total_cases": n,
        "summary": {
            "model_role_accuracy": round(model_correct / n, 4),
            "rule_role_accuracy": round(rule_correct / n, 4),
            "role_agreement_rate": round(agreements / n, 4),
            "mean_model_latency_ms": round(avg_model_lat, 2),
            "mean_rule_latency_ms": round(avg_rule_lat, 2),
            "malformed_output_count": sum(r["model_agent"]["trust_status"] == "CONFIRMED_FAILURE" for r in results),
            "validation_failure_count": sum(r["model_agent"]["validation_score"] == 0.0 for r in results),
        },
        "cases": results,
    }


def save_model_evaluation(results: dict, output_path: str | Path) -> Path:
    """Save model evaluation records exclusively to prevent overwriting existing files."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, allow_nan=False)
    return path
