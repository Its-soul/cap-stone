"""Offline contract tests for Phase 4 role-isolated model specialists and 5-class fault matrix.

Uses explicitly labeled offline fake clients (`FakeRoleSpecialistClient`) so CI runs
deterministically and offline with zero API keys or external network calls.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from cert_recovery.adjudication.domain import ROLES, InsuranceClaim
from cert_recovery.adjudication.independent_verifier import verify_composition_independently
from cert_recovery.adjudication.ledger import Ed25519EventSigner
from cert_recovery.adjudication.model_agent import (
    BudgetExceededError,
    HostedChatModelClient,
    LiveCallBudget,
    ModelAgentAdapter,
    build_phase4_role_backends,
    build_role_prompt,
    make_specialist_profile,
    shared_model_correlation_risk_note,
    validate_specialist_model_id,
)
from cert_recovery.adjudication.phase4_runner import (
    PHASE4_DEV_CASES,
    FakeRoleSpecialistClient,
    execute_phase4_suite,
    run_5class_fault_matrix,
)
from cert_recovery.adjudication.provenance_verifier import verify_persisted_run
from cert_recovery.adjudication.workflow import SpecialistCompositionAdjudication

pytestmark = pytest.mark.adjudication


def test_sub_7b_model_prohibited_for_phase4_specialists():
    """<7B models (including Qwen2.5-0.5B) are rejected for Phase 4 main specialists."""
    for bad in ("onnx-community/Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B", "llama-3.2-3b-instruct", "phi-2"):
        with pytest.raises(ValueError, match="<7B"):
            validate_specialist_model_id(bad)
        with pytest.raises(ValueError, match="<7B"):
            make_specialist_profile("policy", provider="openai_compatible", model_id=bad)

    for ok in ("gemini-2.5-flash", "llama-3.3-70b-versatile", "gpt-4o-mini", "qwen2.5-7b-instruct"):
        validate_specialist_model_id(ok)


def test_strict_role_input_scoping_blocks_foreign_and_oracle_fields():
    """Each specialist prompt receives strictly its own role's fact slice and never foreign/oracle fields."""
    pol_facts = InsuranceClaim().facts("policy")
    binding = {"artifact_id": "facts:policy", "version": 1, "content_hash": "a" * 64}
    msgs = build_role_prompt("policy", pol_facts, binding)
    assert "hospitalization_covered" in msgs[1]["content"]
    assert "documents_available" not in msgs[1]["content"]

    # Injecting a foreign role key or unexpected oracle field raises ValueError
    with pytest.raises(ValueError, match="out-of-scope"):
        build_role_prompt("policy", {**pol_facts, "documents_available": True}, binding)
    with pytest.raises(ValueError, match="out-of-scope"):
        build_role_prompt("policy", {**pol_facts, "expected_label": "APPROVE"}, binding)


def test_four_role_isolated_calls_metadata_and_phase2_phase3_verification(tmp_path: Path):
    """All four roles execute as separate isolated calls verified by Phase 2 & Phase 3."""
    budget = LiveCallBudget(max_calls=20, max_cost_usd=1.0)
    fake_client = FakeRoleSpecialistClient()
    backends = build_phase4_role_backends(
        fake_client,
        provider="offline_fake_contract",
        model_id="gemini-2.5-flash",
        backoff_seconds=0.0,
        budget=budget,
    )
    corr = shared_model_correlation_risk_note(backends)
    assert corr["all_roles_share_single_model"] is True
    assert corr["correlated_failure_risk"].startswith("HIGH_CORRELATED_RISK")

    signer = Ed25519EventSigner.generate()
    prov_dir = tmp_path / "run_clean"
    wf = SpecialistCompositionAdjudication(
        InsuranceClaim(),
        backends=backends,
        signer=signer,
        provenance_dir=prov_dir,
        run_id="contract-4roles",
    )
    res = wf.result()
    assert res["decision"] == "APPROVE"
    assert len(fake_client.call_log) == 4
    assert [c["role"] for c in fake_client.call_log] == list(ROLES)
    assert budget.calls_used == 4
    assert budget.total_tokens == 4 * 208

    for role in ROLES:
        node = wf.system.latest(f"{role}_agent")
        meta = node.payload["raw_output"]["model_metadata"]
        assert meta["provider"] == "offline_fake_contract"
        assert meta["model_id"] == "gemini-2.5-flash"
        assert len(meta["prompt_hash"]) == 64
        assert meta["input_slice_ref"] == f"facts:{role}@1"
        assert meta["usage"]["total_tokens"] == 208
        assert meta["parsed_json"]["decision"] == "APPROVE"

    round_ref = wf.system.latest("round").ref
    indep = verify_composition_independently(
        wf.system.artifact(round_ref),
        wf.system.provenance(round_ref),
        wf.ledger,
    )
    assert indep.passed is True, indep.reason
    pvr = verify_persisted_run(prov_dir / "events.jsonl", signer.public_key, prov_dir / "checkpoint.json")
    assert pvr.valid is True and pvr.checkpoint_verified is True


def test_hard_budget_cap_stops_immediately_before_overspend(tmp_path: Path):
    """Hard call/USD budget cap aborts workflow immediately before exceeding limit."""
    budget = LiveCallBudget(max_calls=2, max_cost_usd=1.0)
    fake_client = FakeRoleSpecialistClient()
    backends = build_phase4_role_backends(
        fake_client,
        provider="offline_fake_contract",
        model_id="gpt-4o-mini",
        backoff_seconds=0.0,
        budget=budget,
    )
    with pytest.raises(BudgetExceededError, match="Call budget cap reached"):
        SpecialistCompositionAdjudication(InsuranceClaim(), backends=backends)
    assert len(fake_client.call_log) == 2

    usd_budget = LiveCallBudget(max_calls=10, max_cost_usd=0.01)
    usd_budget.record(prompt_tokens=100, completion_tokens=50, estimated_cost_usd=0.01)
    with pytest.raises(BudgetExceededError, match="Cost budget cap reached"):
        usd_budget.check_available()


def test_bounded_retry_recovers_on_transient_failure():
    """Crash/timeout retries up to max_retries=2 and succeeds if transient fault clears."""
    attempts = {"n": 0}
    inner = FakeRoleSpecialistClient()

    def flaky_client(messages, profile):
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise TimeoutError("Transient timeout")
        return inner(messages, profile)

    adapter = ModelAgentAdapter(
        make_specialist_profile(
            "evidence",
            provider="offline_fake_contract",
            model_id="gemini-2.5-flash",
            max_retries=2,
            backoff_seconds=0.0,
        ),
        client=flaky_client,
    )
    res = adapter.execute(
        "evidence",
        "Check documents",
        InsuranceClaim().facts("evidence"),
        {"artifact_id": "facts:evidence", "version": 1, "content_hash": "b" * 64},
    )
    assert attempts["n"] == 3
    assert res["decision"] == "APPROVE"
    assert res["model_metadata"]["retry_count"] == 2
    assert res["model_metadata"]["call_status"] == "SUCCESS"


def test_five_class_fault_matrix_all_scenarios_pass(tmp_path: Path):
    """All 5 fault classes run end-to-end through SpecialistCompositionAdjudication cleanly."""
    rows = run_5class_fault_matrix(
        base_dir=tmp_path / "faults",
        client=FakeRoleSpecialistClient(),
        provider="offline_fake_contract",
        model_id="llama-3.3-70b-versatile",
        fault_source_label="OFFLINE_LABELED_FAKE_CONTRACT_ONLY",
        backoff_seconds=0.0,
    )
    classes = {r["fault_class"] for r in rows}
    assert classes == {
        "crash_timeout",
        "malformed_output",
        "stale_binding",
        "valid_but_wrong",
        "adversarial_specialist",
    }
    assert all(r["status"] == "PASS" for r in rows), rows
    # Confirm F04b unfaithful reasoning acceptance is surfaced transparently (never hidden)
    f04b = next(r for r in rows if r["scenario_id"] == "F04b_valid_json_unfaithful_reasoning")
    assert f04b["validator_outcome"] == "ACCEPTED_UNVERIFIED_REASONING"
    assert f04b["actual_decision"] == "APPROVE"


def test_phase4_suite_reports_blocked_when_live_key_missing_and_never_fakes_live(tmp_path: Path):
    """Without authorized live API config/key, live execution reports BLOCKED and never emits fake live logs."""
    assert len(PHASE4_DEV_CASES) == 6
    summary = execute_phase4_suite(
        tmp_path / "phase4_blocked",
        provider="gemini",
        model_id="gemini-2.5-flash",
        api_key_env_var="NONEXISTENT_PHASE4_TEST_KEY",
    )
    assert summary["live_execution"]["status"] == "BLOCKED"
    assert summary["live_execution"]["live_pilot_calls_path"] is None
    assert summary["live_execution"]["budget"]["calls_used"] == 0
    assert not (tmp_path / "phase4_blocked" / "live_pilot_calls.jsonl").exists()

    off = summary["offline_contract_verification"]
    assert off["status"] == "PASS"
    assert off["nontrivial_non_review_decisions_present"] is True
    assert off["decision_distribution"]["APPROVE"] >= 1
    assert off["decision_distribution"]["REJECT"] >= 1
    assert off["decision_distribution"]["REVIEW"] >= 1
    assert off["fault_scenarios_passed"] == off["fault_scenarios_total"]

    # Also verify HostedChatModelClient reports unconfigured when env var is missing
    hc = HostedChatModelClient(provider="gemini", api_key_env_var="NONEXISTENT_PHASE4_TEST_KEY")
    assert hc.is_configured() is False


def test_gpt_oss_20b_and_120b_accepted_by_model_policy():
    """Verify openai/gpt-oss-20b and openai/gpt-oss-120b pass the >=7B Phase 4 model gate."""
    for model_id in ("openai/gpt-oss-20b", "openai/gpt-oss-120b"):
        validate_specialist_model_id(model_id)
        prof = make_specialist_profile("policy", provider="groq", model_id=model_id)
        assert prof.model_id == model_id
        assert prof.provider == "groq"


def test_groq_provider_defaults_base_url_env_var_and_user_agent(monkeypatch):
    """Verify groq defaults to Groq endpoint + GROQ_API_KEY + explicit User-Agent header."""
    import io
    from cert_recovery.adjudication.model_agent import DEFAULT_USER_AGENT

    client = HostedChatModelClient(provider="groq")
    assert client.base_url == "https://api.groq.com/openai/v1/chat/completions"
    assert client.api_key_env_var == "GROQ_API_KEY"

    custom = HostedChatModelClient(
        provider="openai_compatible",
        api_key_env_var="CUSTOM_KEY",
        base_url="https://example.invalid/v1/chat/completions",
    )
    assert custom.base_url == "https://example.invalid/v1/chat/completions"

    captured: dict[str, object] = {}

    class _FakeResp:
        def __init__(self, payload: bytes):
            self._buf = io.BytesIO(payload)

        def read(self) -> bytes:
            return self._buf.read()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(req, timeout=30.0):
        captured["url"] = req.full_url
        captured["headers"] = {k.lower(): v for k, v in req.header_items()}
        body = {
            "id": "chatcmpl-test",
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "decision": "APPROVE",
                        "confidence": 0.95,
                        "evidence": InsuranceClaim().facts("policy"),
                        "rule_references": ["coverage-v1"],
                        "reasoning_summary": "Policy satisfied.",
                    })
                }
            }],
            "usage": {"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75},
        }
        return _FakeResp(json.dumps(body).encode("utf-8"))

    monkeypatch.setenv("GROQ_API_KEY", "dummy-test-value-not-real")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    prof = make_specialist_profile("policy", provider="groq", model_id="openai/gpt-oss-20b")
    adapter = ModelAgentAdapter(prof, client=client)
    res = adapter.execute(
        "policy",
        "Check policy",
        InsuranceClaim().facts("policy"),
        {"artifact_id": "facts:policy", "version": 1, "content_hash": "a" * 64},
    )
    assert res["decision"] == "APPROVE"
    assert captured["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert captured["headers"]["user-agent"] == DEFAULT_USER_AGENT
    assert captured["headers"]["user-agent"] == "cert-recovery-capstone/1.0"


def test_http_401_404_fail_fast_without_retries_while_429_503_retry(monkeypatch):
    """Non-transient 401/404 errors fail fast with 0 retries; transient 429 retries boundedly."""
    import io
    import urllib.error

    monkeypatch.setenv("GROQ_API_KEY", "dummy-test-value-not-real")
    client = HostedChatModelClient(provider="groq")
    facts = InsuranceClaim().facts("policy")
    binding = {"artifact_id": "facts:policy", "version": 1, "content_hash": "a" * 64}

    for status_code, body_txt in ((401, '{"error":"invalid_api_key"}'), (404, '{"error":"model_not_found"}')):
        calls = {"n": 0}
        budget = LiveCallBudget(max_calls=10, max_cost_usd=1.0)

        def fail_http(req, timeout=30.0, code=status_code, msg=body_txt):
            calls["n"] += 1
            raise urllib.error.HTTPError(
                req.full_url,
                code,
                "Err",
                {},
                io.BytesIO(msg.encode("utf-8")),
            )

        monkeypatch.setattr("urllib.request.urlopen", fail_http)
        adapter = ModelAgentAdapter(
            make_specialist_profile(
                "policy",
                provider="groq",
                model_id="openai/gpt-oss-20b",
                max_retries=2,
                backoff_seconds=0.0,
            ),
            client=client,
            budget=budget,
        )
        res = adapter.execute("policy", "Check policy", facts, binding)
        assert calls["n"] == 1
        assert budget.calls_used == 1
        assert res["model_metadata"]["retry_count"] == 0
        assert res["model_metadata"]["call_status"] == "FAILED"
        assert f"HTTP {status_code}" in res["execution_error"]

    # Contrast with transient HTTP 429: must retry up to max_retries=2 (3 total attempts)
    transient_calls = {"n": 0}

    def fail_429(req, timeout=30.0):
        transient_calls["n"] += 1
        raise urllib.error.HTTPError(
            req.full_url, 429, "Rate limit", {}, io.BytesIO(b'{"error":"rate_limit"}')
        )

    monkeypatch.setattr("urllib.request.urlopen", fail_429)
    budget_429 = LiveCallBudget(max_calls=10, max_cost_usd=1.0)
    adapter_429 = ModelAgentAdapter(
        make_specialist_profile(
            "policy",
            provider="groq",
            model_id="openai/gpt-oss-20b",
            max_retries=2,
            backoff_seconds=0.0,
        ),
        client=client,
        budget=budget_429,
    )
    res_429 = adapter_429.execute("policy", "Check policy", facts, binding)
    assert transient_calls["n"] == 3
    assert budget_429.calls_used == 3
    assert res_429["model_metadata"]["retry_count"] == 2
    assert "HTTP 429" in res_429["execution_error"]


def test_http_429_server_hint_parsing_and_bounded_backoff(monkeypatch):
    """Verify Retry-After header and Groq 'try again in <val>(ms|s)' wait hints with +0.5s buffer and 30s cap."""
    import io
    import urllib.error
    from cert_recovery.adjudication.model_agent import parse_retry_after_hint

    assert parse_retry_after_hint(None, "Please try again in 1.5s") == pytest.approx(1.5)
    assert parse_retry_after_hint(None, "Please try again in 592.5ms.") == pytest.approx(0.5925)
    assert parse_retry_after_hint({"Retry-After": "4"}, "") == pytest.approx(4.0)
    assert parse_retry_after_hint({"retry-after": "2.5"}, "try again in 500ms") == pytest.approx(2.5)

    sleeps: list[float] = []
    monkeypatch.setattr("cert_recovery.adjudication.model_agent.time.sleep", lambda s: sleeps.append(s))
    monkeypatch.setenv("GROQ_API_KEY", "dummy-test-value-not-real")

    facts = InsuranceClaim().facts("policy")
    binding = {"artifact_id": "facts:policy", "version": 1, "content_hash": "a" * 64}
    client = HostedChatModelClient(provider="groq", min_call_interval_sec=0.0)

    seq = {"n": 0}

    def fake_urlopen(req, timeout=30.0):
        seq["n"] += 1
        if seq["n"] == 1:
            raise urllib.error.HTTPError(
                req.full_url,
                429,
                "Too Many Requests",
                {},
                io.BytesIO(b'{"error":{"message":"Rate limit reached. Please try again in 1.5s."}}'),
            )
        if seq["n"] == 2:
            raise urllib.error.HTTPError(
                req.full_url,
                429,
                "Too Many Requests",
                {},
                io.BytesIO(b'{"error":{"message":"Please try again in 592.5ms."}}'),
            )
        # Third attempt succeeds
        class _OkResp:
            def read(self):
                return json.dumps({
                    "choices": [{
                        "message": {
                            "content": json.dumps({
                                "decision": "APPROVE",
                                "confidence": 0.95,
                                "evidence": facts,
                                "rule_references": ["coverage-v1"],
                                "reasoning_summary": "Policy OK after rate limit.",
                            })
                        }
                    }],
                    "usage": {"prompt_tokens": 40, "completion_tokens": 20, "total_tokens": 60},
                }).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        return _OkResp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    budget = LiveCallBudget(max_calls=10, max_cost_usd=1.0)
    adapter = ModelAgentAdapter(
        make_specialist_profile(
            "policy",
            provider="groq",
            model_id="openai/gpt-oss-20b",
            max_retries=2,
            backoff_seconds=0.0,
        ),
        client=client,
        budget=budget,
    )
    res = adapter.execute("policy", "Check policy", facts, binding)
    assert res["decision"] == "APPROVE"
    assert res["model_metadata"]["retry_count"] == 2
    assert budget.calls_used == 3  # All 3 HTTP attempts recorded truthfully
    assert sleeps == [pytest.approx(2.0), pytest.approx(1.0925)]

    # Verify Retry-After header parsing & 30.0s cap on oversized hints
    sleeps.clear()

    def fake_header_429(req, timeout=30.0):
        if not sleeps:
            raise urllib.error.HTTPError(
                req.full_url, 429, "Too Many Requests", {"Retry-After": "2"}, io.BytesIO(b'{"error":"quota"}')
            )
        raise urllib.error.HTTPError(
            req.full_url, 429, "Too Many Requests", {"Retry-After": "120"}, io.BytesIO(b'{"error":"quota"}')
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_header_429)
    res_cap = ModelAgentAdapter(
        make_specialist_profile("policy", provider="groq", model_id="openai/gpt-oss-20b", max_retries=2, backoff_seconds=0.0),
        client=client,
    ).execute("policy", "Check policy", facts, binding)
    assert res_cap["model_metadata"]["retry_count"] == 2
    assert sleeps == [pytest.approx(2.5), pytest.approx(30.0)]


def test_min_call_interval_enforces_spacing_and_cli_flag(monkeypatch):
    """Verify min_call_interval_sec sleeps between consecutive calls and CLI exposes --min-call-interval-sec."""
    import io
    from scripts.run_phase4_pilot import parse_args

    args = parse_args(["--provider", "groq", "--model-id", "openai/gpt-oss-20b", "--min-call-interval-sec", "1.25"])
    assert args.min_call_interval_sec == 1.25

    now = {"t": 100.0}
    sleeps: list[float] = []

    def fake_monotonic():
        return now["t"]

    def fake_sleep(sec: float):
        sleeps.append(sec)
        now["t"] += sec

    monkeypatch.setattr("cert_recovery.adjudication.model_agent.time.monotonic", fake_monotonic)
    monkeypatch.setattr("cert_recovery.adjudication.model_agent.time.sleep", fake_sleep)
    monkeypatch.setenv("GROQ_API_KEY", "dummy-test-value-not-real")

    facts = InsuranceClaim().facts("policy")
    binding = {"artifact_id": "facts:policy", "version": 1, "content_hash": "a" * 64}

    class _OkResp:
        def read(self):
            return json.dumps({
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "decision": "APPROVE",
                            "confidence": 0.95,
                            "evidence": facts,
                            "rule_references": ["coverage-v1"],
                            "reasoning_summary": "Policy OK.",
                        })
                    }
                }],
                "usage": {"prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45},
            }).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=30.0: _OkResp())

    client = HostedChatModelClient(provider="groq", min_call_interval_sec=2.0)
    adapter = ModelAgentAdapter(
        make_specialist_profile("policy", provider="groq", model_id="openai/gpt-oss-20b", backoff_seconds=0.0),
        client=client,
    )
    adapter.execute("policy", "Check policy", facts, binding)
    now["t"] += 0.5  # Only 0.5s elapsed before second call -> expects 1.5s sleep
    adapter.execute("policy", "Check policy", facts, binding)
    now["t"] += 3.0  # 3.0s elapsed (> 2.0s) -> expects no sleep on third call
    adapter.execute("policy", "Check policy", facts, binding)

    assert sleeps == [pytest.approx(1.5)]
