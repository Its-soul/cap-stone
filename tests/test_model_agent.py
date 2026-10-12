"""Behavioral tests for ModelAgentAdapter and evaluation pipeline."""
import json
import pytest

from cert_recovery.adjudication.consensus import assess
from cert_recovery.adjudication.domain import InsuranceClaim, ROLES
from cert_recovery.adjudication.model_agent import (
    ModelAgentAdapter,
    ModelProfile,
    build_compliance_prompt,
    parse_model_response,
)
from cert_recovery.adjudication.model_evaluation import (
    compliance_evaluation_cases,
    mock_model_client,
    run_model_comparison,
    save_model_evaluation,
)
from cert_recovery.adjudication.workflow import InsuranceAdjudication


def test_build_compliance_prompt_structure():
    facts = {"identity_verified": True, "consent_available": True}
    binding = {"artifact_id": "facts:compliance", "version": 1, "content_hash": "abc"}
    messages = build_compliance_prompt(facts, binding)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "identity-consent-v1" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "identity_verified" in messages[1]["content"]


def test_parse_model_response_handles_markdown_and_raw_json():
    raw_json = '{"decision": "APPROVE", "confidence": 0.95}'
    assert parse_model_response(raw_json)["decision"] == "APPROVE"

    markdown_json = '```json\n{"decision": "REVIEW", "confidence": 0.60}\n```'
    assert parse_model_response(markdown_json)["decision"] == "REVIEW"

    wrapped = 'Here is the result:\n```\n{"decision": "APPROVE"}\n```\nThank you.'
    assert parse_model_response(wrapped)["decision"] == "APPROVE"

    with pytest.raises(json.JSONDecodeError):
        parse_model_response("This is not JSON at all.")


def test_model_adapter_valid_execution():
    facts = {"identity_verified": True, "consent_available": True}
    binding = {"artifact_id": "facts:compliance", "version": 1, "content_hash": "abc"}

    adapter = ModelAgentAdapter(client=mock_model_client)
    res = adapter.execute("compliance", "Check compliance", facts, binding)

    assert res["agent_id"] == "compliance_agent"
    assert res["role"] == "compliance"
    assert res["decision"] == "APPROVE"
    assert 0.0 <= res["confidence"] <= 1.0
    assert res["uncertainty"]["confidence_origin"] == "self_reported"
    assert res["backend"].startswith("model-backed-")
    assert res["model_metadata"]["call_status"] == "SUCCESS"

    # Assess with verifier
    assessment = assess(res, "compliance", facts, binding)
    assert assessment.trust_status == "VALIDATED"
    assert assessment.validation_score == 1.0


def test_model_adapter_unsupported_role_raises_error():
    adapter = ModelAgentAdapter(client=mock_model_client)
    with pytest.raises(ValueError, match="configured for compliance"):
        adapter.execute("policy", "Check policy", {}, {})


def test_model_adapter_malformed_output_contained():
    def malformed_client(messages, profile):
        return "Not valid JSON output from LLM."

    adapter = ModelAgentAdapter(client=malformed_client)
    res = adapter.execute("compliance", "Check compliance", {"identity_verified": True}, {})

    assert "execution_error" in res
    assert res["model_metadata"]["call_status"] == "MALFORMED_OUTPUT"

    # Verifier catches it
    assessment = assess(res, "compliance", {"identity_verified": True}, {})
    assert assessment.trust_status == "CONFIRMED_FAILURE"
    assert assessment.validation_score == 0.0


def test_model_adapter_network_timeout_and_retry():
    call_counts = 0

    def timeout_client(messages, profile):
        nonlocal call_counts
        call_counts += 1
        raise TimeoutError("Model endpoint connection timed out")

    adapter = ModelAgentAdapter(client=timeout_client)
    res = adapter.execute("compliance", "Check compliance", {}, {})

    # Single retry policy: attempt 0 + attempt 1 = 2 calls
    assert call_counts == 2
    assert "TimeoutError" in res["execution_error"]
    assert res["model_metadata"]["call_status"] == "FAILED"


def test_model_adapter_no_backend_reports_not_run():
    # Adapter initialized with no client reports NOT_RUN_NO_BACKEND
    adapter = ModelAgentAdapter(client=None)
    res = adapter.execute("compliance", "Check compliance", {}, {})

    assert "execution_error" in res
    assert res["model_metadata"]["call_status"] == "NOT_RUN_NO_BACKEND"


def test_workflow_integration_with_model_backed_agent():
    """Whole claim workflow seamlessly accepts ModelAgentAdapter."""
    claim = InsuranceClaim()
    adapter = ModelAgentAdapter(client=mock_model_client)

    workflow = InsuranceAdjudication(claim, backends={"compliance": adapter})
    final = workflow.result()

    assert final["decision"] == "APPROVE"
    assert final["gate"] == "PASS"

    compliance_artifact = workflow.system.latest("compliance_agent")
    raw_compliance = compliance_artifact.payload["raw_output"]
    assert raw_compliance["backend"].startswith("model-backed-")
    assert raw_compliance["uncertainty"]["confidence_origin"] == "self_reported"


def test_model_evaluation_cases_and_comparison_run(tmp_path):
    cases = compliance_evaluation_cases()
    assert len(cases) == 11

    results = run_model_comparison()
    assert results["total_cases"] == 11
    assert results["summary"]["model_role_accuracy"] == 1.0
    assert results["summary"]["rule_role_accuracy"] == 1.0
    assert results["summary"]["role_agreement_rate"] == 1.0

    target = tmp_path / "model_eval_test.json"
    save_model_evaluation(results, target)
    assert target.exists()
    with pytest.raises(FileExistsError):
        save_model_evaluation(results, target)


def test_build_compliance_prompt_v2_few_shot_structure():
    """Candidate prompt v2 has 2 few-shot demonstrations, concise task, and no distracting metadata."""
    from cert_recovery.adjudication.model_agent import (
        build_compliance_prompt_v1,
        build_compliance_prompt_v2,
    )
    facts = {"consent_available": True, "identity_verified": False}
    binding = {"artifact_id": "facts:compliance", "version": 1, "content_hash": "abc"}

    v1_msgs = build_compliance_prompt_v1(facts, binding)
    v2_msgs = build_compliance_prompt_v2(facts, binding)

    # v1 has 2 messages (system, user)
    assert len(v1_msgs) == 2
    assert "Input Reference:" in v1_msgs[1]["content"]

    # v2 has 6 messages: system, demo1 user, demo1 assistant, demo2 user, demo2 assistant, query user
    assert len(v2_msgs) == 6
    assert v2_msgs[0]["role"] == "system"
    assert "copy the exact boolean values from \"Facts\" faithfully" in v2_msgs[0]["content"]

    # Demonstrations check
    assert v2_msgs[1]["role"] == "user"
    assert "Facts:" in v2_msgs[1]["content"]
    assert v2_msgs[2]["role"] == "assistant"
    demo1_out = json.loads(v2_msgs[2]["content"])
    assert demo1_out["decision"] == "APPROVE"
    assert demo1_out["confidence"] == 1.0
    assert demo1_out["evidence"] == {"consent_available": True, "identity_verified": True}

    assert v2_msgs[3]["role"] == "user"
    assert v2_msgs[4]["role"] == "assistant"
    demo2_out = json.loads(v2_msgs[4]["content"])
    assert demo2_out["decision"] == "REVIEW"
    assert demo2_out["confidence"] == 0.0
    assert demo2_out["evidence"] == {"consent_available": False, "identity_verified": True}

    # Target user message does not contain distracting metadata
    assert v2_msgs[5]["role"] == "user"
    assert "Input Reference" not in v2_msgs[5]["content"]
    assert "Facts: {\"consent_available\": true, \"identity_verified\": false}" in v2_msgs[5]["content"]


def test_model_adapter_candidate_prompt_execution():
    """ModelAgentAdapter executes cleanly with prompt_version='v2'."""
    facts = {"identity_verified": True, "consent_available": True}
    binding = {"artifact_id": "facts:compliance", "version": 1, "content_hash": "abc"}

    profile_v2 = ModelProfile(prompt_version="v2")
    adapter = ModelAgentAdapter(profile=profile_v2, client=mock_model_client)
    res = adapter.execute("compliance", "Check compliance", facts, binding)

    assert res["model_metadata"]["prompt_version"] == "v2"
    assert res["model_metadata"]["call_status"] == "SUCCESS"
    assert res["decision"] == "APPROVE"
    assert res["input_reference"] == binding
