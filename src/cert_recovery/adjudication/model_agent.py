"""Model-backed agent adapter implementing the AgentBackend protocol.

Supports:
- Strict role-isolated prompting across all four adjudication roles
  (`policy`, `evidence`, `risk`, `compliance`) with positive input-slice scoping.
- Hosted API and >=7B local specialist profiles (explicitly prohibiting <7B
  models such as Qwen2.5-0.5B for main Phase 4 specialist runs).
- Typed JSON decoding, bounded retries with backoff, hard call/USD budget caps,
  and full execution provenance (provider, model_id, prompt SHA-256 hash,
  input slice ref, latency, token counts, retry count, raw & parsed JSON).
- Explicit documentation of shared-model correlated failure risk.
"""
from dataclasses import asdict, dataclass
import json
import os
import re
import time
from typing import Any, Callable
import urllib.error
import urllib.request

from ..crypto import canonical_json, sha256
from .domain import DECISIONS, ROLES, RULES, AgentBackend, rule_decision

FROZEN_MODEL_ID = "onnx-community/Qwen2.5-0.5B-Instruct"
FROZEN_REVISION = "22942cb7d7ba4cc81bb4673549ca4d4614469b5e"
DEFAULT_FILE_NAME = "model_int8.onnx"
SUPPORTED_ROLE = "compliance"
DEFAULT_TIMEOUT_SECONDS = 60.0
MAX_RETRIES = 1
TEMPERATURE = 0.0
MAX_TOKENS = 128

_SUB_7B_RE = re.compile(
    r"(?:^|[-_/:])(?:0\.[0-9]+|1(?:\.[0-9]+)?|2(?:\.[0-9]+)?|3(?:\.[0-9]+)?|4(?:\.[0-9]+)?|5(?:\.[0-9]+)?|6(?:\.[0-9]+)?)b(?:$|[-_/:@])",
    re.IGNORECASE,
)
_TINY_MODEL_MARKERS = ("qwen2.5-0.5b", "tinyllama", "smollm", "phi-2", "phi-1")


class BudgetExceededError(RuntimeError):
    """Raised before making a model call when Phase 4 call or USD budget would be exceeded."""


DEFAULT_USER_AGENT = "cert-recovery-capstone/1.0"
NON_RETRYABLE_HTTP_STATUS_CODES = frozenset({400, 401, 403, 404})


class NonTransientHTTPError(RuntimeError):
    """Raised for non-transient HTTP 400/401/403/404 responses that must not be retried."""

    def __init__(self, status_code: int, body_snippet: str = ""):
        self.status_code = int(status_code)
        self.body_snippet = body_snippet
        msg = f"HTTP {self.status_code}"
        if body_snippet:
            msg = f"{msg}: {body_snippet}"
        super().__init__(msg)


_RETRY_IN_BODY_RE = re.compile(
    r"try\s+again\s+in\s+(\d+(?:\.\d+)?)\s*(ms|s)\b",
    re.IGNORECASE,
)


def parse_retry_after_hint(headers: Any = None, body_text: str = "") -> float | None:
    """Extract wait duration in seconds from Retry-After header or provider error body text."""
    candidates: list[float] = []
    if headers is not None:
        raw_header = None
        if hasattr(headers, "items"):
            for k, v in headers.items():
                if str(k).lower() == "retry-after":
                    raw_header = str(v).strip()
                    break
        elif hasattr(headers, "get"):
            val = headers.get("Retry-After") or headers.get("retry-after")
            if val is not None:
                raw_header = str(val).strip()
        if raw_header:
            try:
                sec = float(raw_header)
                if sec >= 0.0:
                    candidates.append(sec)
            except ValueError:
                pass
    if body_text:
        m = _RETRY_IN_BODY_RE.search(body_text)
        if m:
            value = float(m.group(1))
            unit = m.group(2).lower()
            candidates.append(value / 1000.0 if unit == "ms" else value)
    return max(candidates) if candidates else None


class TransientHTTPError(RuntimeError):
    """Raised for transient HTTP 429/5xx errors carrying any parsed retry delay hint."""

    def __init__(
        self,
        status_code: int,
        body_snippet: str = "",
        retry_after_seconds: float | None = None,
    ):
        self.status_code = int(status_code)
        self.body_snippet = body_snippet
        self.retry_after_seconds = retry_after_seconds
        msg = f"HTTP {self.status_code}"
        if body_snippet:
            msg = f"{msg}: {body_snippet}"
        super().__init__(msg)


@dataclass
class LiveCallBudget:
    """Hard call-count and estimated-USD cap for model-backed execution."""

    max_calls: int = 60
    max_cost_usd: float = 1.0
    calls_used: int = 0
    cost_usd_used: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def check_available(self) -> None:
        if self.calls_used >= self.max_calls:
            raise BudgetExceededError(
                f"Call budget cap reached: {self.calls_used} >= max_calls={self.max_calls}"
            )
        if self.cost_usd_used >= self.max_cost_usd:
            raise BudgetExceededError(
                f"Cost budget cap reached: ${self.cost_usd_used:.6f} >= max_cost_usd=${self.max_cost_usd:.2f}"
            )

    def record(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int | None = None,
        estimated_cost_usd: float = 0.0,
    ) -> None:
        self.calls_used += 1
        self.prompt_tokens += max(0, int(prompt_tokens or 0))
        self.completion_tokens += max(0, int(completion_tokens or 0))
        tot = int(total_tokens) if total_tokens is not None else int(prompt_tokens or 0) + int(completion_tokens or 0)
        self.total_tokens += max(0, tot)
        self.cost_usd_used += max(0.0, float(estimated_cost_usd or 0.0))

    def snapshot(self) -> dict[str, Any]:
        return {
            "max_calls": self.max_calls,
            "max_cost_usd": self.max_cost_usd,
            "calls_used": self.calls_used,
            "cost_usd_used": round(self.cost_usd_used, 6),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class ModelClientResponse:
    """Optional structured response from a model client carrying raw text and token usage."""

    raw_text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float = 0.0
    provider_response_id: str | None = None


def is_prohibited_small_model(model_id: str) -> bool:
    """Return True if `model_id` identifies a <7B model prohibited for Phase 4 main specialists."""
    lower = model_id.strip().lower()
    if any(tok in lower for tok in _TINY_MODEL_MARKERS):
        return True
    return bool(_SUB_7B_RE.search(lower))


def validate_specialist_model_id(model_id: str, *, allow_small_legacy: bool = False) -> None:
    """Enforce Phase 4 >=7B / hosted API model policy."""
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("Model ID must be a non-empty string")
    if not allow_small_legacy and is_prohibited_small_model(model_id):
        raise ValueError(
            f"Model '{model_id}' appears to be <7B and is prohibited for Phase 4 main specialists; "
            "configure a hosted API model or >=7B model."
        )


def shared_model_correlation_risk_note(backends: dict[str, Any]) -> dict[str, Any]:
    """Summarize whether all specialist roles share a single underlying model family."""
    model_ids = {}
    for role, backend in backends.items():
        if hasattr(backend, "profile"):
            model_ids[role] = f"{backend.profile.provider}:{backend.profile.model_id}"
        else:
            model_ids[role] = type(backend).__name__
    unique = sorted(set(model_ids.values()))
    shared = len(unique) == 1 and len(model_ids) > 1
    return {
        "role_models": model_ids,
        "unique_model_endpoints": unique,
        "all_roles_share_single_model": shared,
        "correlated_failure_risk": (
            "HIGH_CORRELATED_RISK: All four role-isolated prompts call the same underlying model checkpoint/endpoint. "
            "Prompt isolation prevents cross-role context leakage, NOT correlated formatting regressions, outage, or reasoning blind spots."
            if shared
            else "DIVERSE_OR_HYBRID_BACKENDS"
        ),
    }


FAULT_MODEL_SPECIFICATION: tuple[dict[str, str], ...] = (
    {
        "fault_class": "crash_timeout",
        "failure_mode": "Transient network timeout, HTTP 429/5xx, or adapter exception exhausts bounded retries.",
        "containment_rule": "Bounded retry with backoff (max_retries=2 default), then return execution_error -> assess_specialist yields UNCERTAIN (UNK) / REVIEW.",
        "safety_invariant": "Never crashes pipeline and never converts an unavailable model into APPROVE.",
    },
    {
        "fault_class": "malformed_output",
        "failure_mode": "Unparseable text, missing keys, invalid confidence, wrong types, or unknown decision token.",
        "containment_rule": "Do not coerce unknown decisions; fail schema checks cleanly -> assess_specialist yields UNCERTAIN (UNK) / REVIEW (not FAIL).",
        "safety_invariant": "Prevents syntax failure from causing unsafe APPROVE or false substantive REJECT.",
    },
    {
        "fault_class": "stale_binding",
        "failure_mode": "Specialist binds output to an outdated artifact version or mismatched content hash.",
        "containment_rule": "Rejected by assess_specialist (stale_or_unbound_input -> UNK/REVIEW) and refused by verify_composition_independently / commit_decision.",
        "safety_invariant": "Blocks replay of stale specialist conclusions after source fact mutation.",
    },
    {
        "fault_class": "valid_but_wrong",
        "failure_mode": "Syntactically valid typed JSON contradicts deterministic rules or provides unfaithful reasoning.",
        "containment_rule": "Decision or copied-fact contradiction is caught by assess_specialist (decision_conflicts_with_rule / evidence_mismatch -> UNK/REVIEW); semantic reasoning errors with matching labels are explicitly audited as accepted.",
        "safety_invariant": "Contradictory APPROVE is contained; unverified free-text justification limits remain visible.",
    },
    {
        "fault_class": "adversarial_specialist",
        "failure_mode": "One compromised specialist tries to force APPROVE over a true exclusion or fabricates copied evidence fields.",
        "containment_rule": "Host checks authoritative facts/rule in source ancestry; compromised role forced to INVALID/UNK while mandatory exclusion gate still enforces REJECT under substantive_exclusion (REVIEW under strict_audit).",
        "safety_invariant": "Single compromised role cannot force an unsafe APPROVE.",
    },
)


@dataclass(frozen=True)
class ModelProfile:
    model_id: str = FROZEN_MODEL_ID
    revision: str = FROZEN_REVISION
    file_name: str = DEFAULT_FILE_NAME
    device: str = "cpu"
    role: str = SUPPORTED_ROLE
    temperature: float = TEMPERATURE
    max_tokens: int = MAX_TOKENS
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = MAX_RETRIES
    prompt_version: str = "v1"
    provider: str = "legacy_local_onnx"
    backoff_seconds: float = 0.0
    strict_input_scoping: bool = False
    allow_small_legacy: bool = True

    def __post_init__(self):
        if self.role not in ROLES:
            raise ValueError(f"Unknown role: {self.role}")
        validate_specialist_model_id(self.model_id, allow_small_legacy=self.allow_small_legacy)


def make_specialist_profile(
    role: str,
    *,
    provider: str,
    model_id: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
    timeout_seconds: float = 30.0,
    max_retries: int = 2,
    backoff_seconds: float = 0.25,
    prompt_version: str = "v1",
) -> ModelProfile:
    """Create a Phase 4 strictly role-scoped profile (>=7B / hosted API policy enforced)."""
    return ModelProfile(
        model_id=model_id,
        revision="api",
        file_name="api",
        device="remote",
        role=role,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        prompt_version=prompt_version,
        provider=provider,
        backoff_seconds=backoff_seconds,
        strict_input_scoping=True,
        allow_small_legacy=False,
    )


def build_compliance_prompt_v1(facts: dict, binding: dict) -> list[dict[str, str]]:
    """Original baseline prompt (v1): 0-shot abstract schema with input-reference metadata."""
    system_prompt = (
        "You are an insurance compliance agent evaluating healthcare claims.\n"
        "Role: compliance\n"
        "Governing Rule: identity-consent-v1\n"
        "Requirements:\n"
        "1. Check identity verification and consent availability.\n"
        "2. If identity_verified is true AND consent_available is true, return APPROVE.\n"
        "3. If either is false or missing, return REVIEW (never REJECT).\n"
        "4. In rule_references, cite exactly ['identity-consent-v1'].\n"
        "5. Respond ONLY with a valid JSON object matching the requested schema."
    )
    user_prompt = (
        f"Input Reference: {json.dumps(binding, sort_keys=True)}\n"
        f"Facts to Evaluate: {json.dumps(facts, sort_keys=True)}\n\n"
        "Return a JSON object with this exact structure:\n"
        "{\n"
        '  "decision": "APPROVE" or "REVIEW",\n'
        '  "confidence": <float in [0.0, 1.0]>,\n'
        '  "evidence": {"identity_verified": <bool>, "consent_available": <bool>},\n'
        '  "rule_references": ["identity-consent-v1"],\n'
        '  "reasoning_summary": "<concise explanation>"\n'
        "}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_compliance_prompt_v2(facts: dict, binding: dict | None = None) -> list[dict[str, str]]:
    """Candidate prompt (v2): few-shot structured format with explicit evidence fidelity and concise task."""
    system_prompt = (
        "You are an insurance compliance specialist evaluating patient identity and consent.\n"
        "Governing Rule: identity-consent-v1\n\n"
        "Decision Rules:\n"
        "- If identity_verified is true AND consent_available is true -> return \"APPROVE\".\n"
        "- If either fact is false -> return \"REVIEW\". Never return \"REJECT\".\n\n"
        "Output Requirements:\n"
        "Respond ONLY with a valid JSON object matching this exact schema:\n"
        "{\n"
        '  "decision": "APPROVE" or "REVIEW",\n'
        '  "confidence": 1.0 (for APPROVE) or 0.0 (for REVIEW),\n'
        '  "evidence": {"consent_available": <bool>, "identity_verified": <bool>},\n'
        '  "rule_references": ["identity-consent-v1"],\n'
        '  "reasoning_summary": "<concise explanation>"\n'
        "}\n\n"
        "Rules for evidence:\n"
        "In the \"evidence\" object, copy the exact boolean values from \"Facts\" faithfully.\n"
        "Do not invent values. Do not output text outside the JSON object. Do not add extra keys."
    )
    # Demonstration 1: Positive APPROVE case
    demo1_user = 'Facts: {"consent_available": true, "identity_verified": true}'
    demo1_assistant = (
        '{\n'
        '  "decision": "APPROVE",\n'
        '  "confidence": 1.0,\n'
        '  "evidence": {"consent_available": true, "identity_verified": true},\n'
        '  "rule_references": ["identity-consent-v1"],\n'
        '  "reasoning_summary": "Both identity and consent are verified."\n'
        '}'
    )
    # Demonstration 2: Negative REVIEW case
    demo2_user = 'Facts: {"consent_available": false, "identity_verified": true}'
    demo2_assistant = (
        '{\n'
        '  "decision": "REVIEW",\n'
        '  "confidence": 0.0,\n'
        '  "evidence": {"consent_available": false, "identity_verified": true},\n'
        '  "rule_references": ["identity-consent-v1"],\n'
        '  "reasoning_summary": "Consent is missing; manual review required."\n'
        '}'
    )
    facts_json = json.dumps(facts, sort_keys=True)
    target_user = f"Facts: {facts_json}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": demo1_user},
        {"role": "assistant", "content": demo1_assistant},
        {"role": "user", "content": demo2_user},
        {"role": "assistant", "content": demo2_assistant},
        {"role": "user", "content": target_user},
    ]


def build_compliance_prompt(facts: dict, binding: dict, prompt_version: str = "v1") -> list[dict[str, str]]:
    """Construct deterministic chat messages for the compliance role by version."""
    if prompt_version == "v2":
        return build_compliance_prompt_v2(facts, binding)
    return build_compliance_prompt_v1(facts, binding)


_ROLE_PROMPT_SPECS = {
    "policy": {
        "rule": "coverage-v1",
        "decisions": ('"APPROVE"', '"REJECT"'),
        "instructions": (
            "1. Check hospitalization_covered, waiting_period_satisfied, and amount_inr <= coverage_limit_inr.\n"
            "2. If hospitalization_covered is true AND waiting_period_satisfied is true AND amount_inr <= coverage_limit_inr, return APPROVE.\n"
            "3. Otherwise return REJECT (never REVIEW).\n"
            "4. In rule_references, cite exactly [\"coverage-v1\"]."
        ),
        "evidence_schema": '{"amount_inr": <int>, "coverage_limit_inr": <int>, "hospitalization_covered": <bool>, "waiting_period_satisfied": <bool>}',
    },
    "evidence": {
        "rule": "documents-v1",
        "decisions": ('"APPROVE"', '"REVIEW"'),
        "instructions": (
            "1. Check documents_available and invoice_matches.\n"
            "2. If documents_available is true AND invoice_matches is true, return APPROVE.\n"
            "3. Otherwise return REVIEW (never REJECT).\n"
            "4. In rule_references, cite exactly [\"documents-v1\"]."
        ),
        "evidence_schema": '{"documents_available": <bool>, "invoice_matches": <bool>}',
    },
    "risk": {
        "rule": "fraud-screen-v1",
        "decisions": ('"APPROVE"', '"REVIEW"'),
        "instructions": (
            "1. Check duplicate_claim and provider_flagged.\n"
            "2. If duplicate_claim is false AND provider_flagged is false, return APPROVE.\n"
            "3. Otherwise return REVIEW (never REJECT).\n"
            "4. In rule_references, cite exactly [\"fraud-screen-v1\"]."
        ),
        "evidence_schema": '{"duplicate_claim": <bool>, "provider_flagged": <bool>}',
    },
}


def validate_role_scoped_facts(
    role: str,
    facts: dict[str, Any],
    *,
    require_all_role_keys: bool = False,
) -> dict[str, Any]:
    """Enforce strict role input scoping using a positive per-role field whitelist."""
    if role not in ROLES:
        raise ValueError(f"Unknown specialist role: {role}")
    if not isinstance(facts, dict):
        raise ValueError(f"Role '{role}' facts must be a dict")
    allowed_keys = set(RULES[role][1])
    unexpected = sorted(set(facts) - allowed_keys)
    if unexpected:
        raise ValueError(
            f"Role '{role}' received out-of-scope input fields: {unexpected}; permitted: {sorted(allowed_keys)}"
        )
    if require_all_role_keys:
        missing = sorted(allowed_keys - set(facts))
        if missing:
            raise ValueError(f"Role '{role}' missing required input fields: {missing}")
    return {k: facts[k] for k in sorted(facts)}


def build_role_prompt(
    role: str,
    facts: dict[str, Any],
    binding: dict[str, Any],
    prompt_version: str = "v1",
    *,
    require_all_role_keys: bool = True,
) -> list[dict[str, str]]:
    """Construct strict role-isolated chat messages for any of the four specialist roles."""
    scoped_facts = validate_role_scoped_facts(
        role, facts, require_all_role_keys=require_all_role_keys
    )
    if role == "compliance":
        return build_compliance_prompt(scoped_facts, binding, prompt_version=prompt_version)

    spec = _ROLE_PROMPT_SPECS[role]
    decision_str = " or ".join(spec["decisions"])
    system_prompt = (
        f"You are an insurance {role} specialist evaluating healthcare claims.\n"
        f"Role: {role}\n"
        f"Governing Rule: {spec['rule']}\n"
        "Requirements:\n"
        f"{spec['instructions']}\n"
        "5. Copy only the provided scoped facts faithfully into evidence.\n"
        "6. Respond ONLY with a valid JSON object matching the requested schema."
    )
    user_prompt = (
        f"Input Reference: {json.dumps(binding, sort_keys=True)}\n"
        f"Scoped Facts ({role} only): {json.dumps(scoped_facts, sort_keys=True)}\n\n"
        "Return a JSON object with this exact structure:\n"
        "{\n"
        f'  "decision": {decision_str},\n'
        '  "confidence": <float in [0.0, 1.0]>,\n'
        f'  "evidence": {spec["evidence_schema"]},\n'
        f'  "rule_references": ["{spec["rule"]}"],\n'
        '  "reasoning_summary": "<concise explanation>"\n'
        "}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_model_response(raw_text: str) -> dict:
    """Extract and parse JSON from model output, handling markdown blocks."""
    text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    else:
        brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if brace_match:
            text = brace_match.group(1).strip()
    return json.loads(text)


class ModelAgentAdapter:
    """Genuine model-backed agent adapter conforming to AgentBackend protocol."""

    def __init__(
        self,
        profile: ModelProfile | None = None,
        *,
        client: Callable[[list[dict[str, str]], ModelProfile], str | ModelClientResponse | dict[str, Any]] | None = None,
        offline_fallback: bool = False,
        budget: LiveCallBudget | None = None,
    ):
        self.profile = profile or ModelProfile()
        self.client = client
        self.offline_fallback = offline_fallback
        self.budget = budget

    def execute(self, role: str, prompt: str, facts: dict, input_reference: dict) -> dict:
        """Execute role evaluation via model backend or client."""
        if role != self.profile.role:
            raise ValueError(f"Model adapter configured for {self.profile.role}, not {role}")

        scoped_facts = validate_role_scoped_facts(
            role, facts, require_all_role_keys=self.profile.strict_input_scoping
        )
        messages = build_role_prompt(
            role,
            scoped_facts,
            input_reference,
            prompt_version=self.profile.prompt_version,
            require_all_role_keys=self.profile.strict_input_scoping,
        )
        prompt_hash = sha256(messages)
        if (
            isinstance(input_reference, dict)
            and "artifact_id" in input_reference
            and "version" in input_reference
        ):
            input_slice_ref = f"{input_reference['artifact_id']}@{input_reference['version']}"
        else:
            input_slice_ref = canonical_json(input_reference)

        start_time = time.perf_counter()
        raw_text = None
        usage: dict[str, Any] = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
        estimated_cost = 0.0
        call_status = "PENDING"
        error_msg = None
        retry_count = 0

        if self.client is not None:
            for attempt in range(self.profile.max_retries + 1):
                if self.budget is not None:
                    self.budget.check_available()
                try:
                    raw_out = self.client(messages, self.profile)
                    if isinstance(raw_out, ModelClientResponse):
                        raw_text = raw_out.raw_text
                        usage = {
                            "prompt_tokens": raw_out.prompt_tokens,
                            "completion_tokens": raw_out.completion_tokens,
                            "total_tokens": raw_out.total_tokens,
                        }
                        estimated_cost = raw_out.estimated_cost_usd
                    elif isinstance(raw_out, dict) and "raw_text" in raw_out:
                        raw_text = str(raw_out["raw_text"])
                        usage = {
                            "prompt_tokens": raw_out.get("prompt_tokens"),
                            "completion_tokens": raw_out.get("completion_tokens"),
                            "total_tokens": raw_out.get("total_tokens"),
                        }
                        estimated_cost = float(raw_out.get("estimated_cost_usd", 0.0) or 0.0)
                    else:
                        raw_text = raw_out
                    if self.budget is not None:
                        self.budget.record(
                            prompt_tokens=usage["prompt_tokens"] or 0,
                            completion_tokens=usage["completion_tokens"] or 0,
                            total_tokens=usage["total_tokens"],
                            estimated_cost_usd=estimated_cost,
                        )
                    call_status = "SUCCESS"
                    break
                except BudgetExceededError:
                    raise
                except NonTransientHTTPError as e:
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    call_status = "FAILED"
                    if self.budget is not None:
                        self.budget.record(estimated_cost_usd=0.0)
                    break
                except TransientHTTPError as e:
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    call_status = "FAILED"
                    if self.budget is not None:
                        self.budget.record(estimated_cost_usd=0.0)
                    if attempt == self.profile.max_retries:
                        break
                    retry_count += 1
                    if e.retry_after_seconds is not None:
                        wait_sec = min(30.0, e.retry_after_seconds + 0.5)
                        if wait_sec > 0:
                            time.sleep(wait_sec)
                    elif self.profile.backoff_seconds > 0:
                        time.sleep(self.profile.backoff_seconds * (2 ** attempt))
                except urllib.error.HTTPError as e:
                    error_msg = f"HTTPError: HTTP {e.code}"
                    call_status = "FAILED"
                    if self.budget is not None:
                        self.budget.record(estimated_cost_usd=0.0)
                    if e.code in NON_RETRYABLE_HTTP_STATUS_CODES or attempt == self.profile.max_retries:
                        break
                    retry_count += 1
                    hint_sec = parse_retry_after_hint(getattr(e, "headers", None))
                    if e.code == 429 and hint_sec is not None:
                        wait_sec = min(30.0, hint_sec + 0.5)
                        if wait_sec > 0:
                            time.sleep(wait_sec)
                    elif self.profile.backoff_seconds > 0:
                        time.sleep(self.profile.backoff_seconds * (2 ** attempt))
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    call_status = "FAILED"
                    if self.budget is not None:
                        self.budget.record(estimated_cost_usd=0.0)
                    if attempt == self.profile.max_retries:
                        break
                    retry_count += 1
                    if self.profile.backoff_seconds > 0:
                        time.sleep(self.profile.backoff_seconds * (2 ** attempt))
        else:
            # No live model client configured or available
            call_status = "NOT_RUN_NO_BACKEND"
            error_msg = "Live model endpoint unavailable; offline execution policy enforced."

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        base_meta = {
            "provider": self.profile.provider,
            "model_id": self.profile.model_id,
            "revision": self.profile.revision,
            "prompt_version": self.profile.prompt_version,
            "prompt_hash": prompt_hash,
            "input_slice_ref": input_slice_ref,
            "temperature": self.profile.temperature,
            "max_tokens": self.profile.max_tokens,
            "call_status": call_status,
            "latency_ms": latency_ms,
            "retry_count": retry_count,
            "usage": usage,
        }

        if call_status != "SUCCESS" or not raw_text:
            return {
                "agent_id": role + "_agent",
                "role": role,
                "execution_error": error_msg or "Empty model response",
                "backend": f"model-backed-{self.profile.provider}:{self.profile.model_id}",
                "model_metadata": {**base_meta, "raw_response": raw_text},
            }

        # Parse structured output
        try:
            parsed = parse_model_response(raw_text)
            if not isinstance(parsed, dict):
                raise ValueError("Parsed model output must be a JSON object")
            raw_dec = parsed.get("decision")
            decision = raw_dec.upper() if isinstance(raw_dec, str) else raw_dec

            raw_conf = parsed.get("confidence")
            if raw_conf is None:
                confidence = None
                confidence_origin = "omitted"
            else:
                try:
                    confidence = float(raw_conf)
                    if not (0.0 <= confidence <= 1.0):
                        confidence = None
                        confidence_origin = "out_of_bounds"
                    else:
                        confidence_origin = "self_reported"
                except (TypeError, ValueError):
                    confidence = None
                    confidence_origin = "unparseable"

            evidence = parsed.get("evidence")
            rule_refs = parsed.get("rule_references")
            reasoning = parsed.get("reasoning_summary")
            if not isinstance(reasoning, str) or not reasoning.strip():
                reasoning = ""

            return {
                "agent_id": role + "_agent",
                "role": role,
                "decision": decision,
                "confidence": confidence,
                "evidence": evidence,
                "rule_references": rule_refs,
                "reasoning_summary": reasoning,
                "input_reference": input_reference,
                "provenance_reference": None,
                "uncertainty": {
                    "confidence_origin": confidence_origin,
                    "calibration": "uncalibrated",
                    "raw_model_confidence": raw_conf,
                    "latent_entropy_vector": None,
                },
                "backend": f"model-backed-{self.profile.provider}:{self.profile.model_id}",
                "model_metadata": {
                    **base_meta,
                    "raw_response": raw_text,
                    "parsed_json": parsed,
                },
            }
        except Exception as parse_error:
            return {
                "agent_id": role + "_agent",
                "role": role,
                "execution_error": f"ParseError: {str(parse_error)}",
                "raw_text": raw_text,
                "backend": f"model-backed-{self.profile.provider}:{self.profile.model_id}",
                "model_metadata": {
                    **base_meta,
                    "call_status": "MALFORMED_OUTPUT",
                    "raw_response": raw_text,
                },
            }


class LocalOnnxModelClient:
    """Genuine local ONNX Runtime inference client for Qwen2.5-0.5B-Instruct."""

    def __init__(
        self,
        model_id: str = FROZEN_MODEL_ID,
        revision: str = FROZEN_REVISION,
        file_name: str = DEFAULT_FILE_NAME,
        device: str = "cpu",
    ):
        self.model_id = model_id
        self.revision = revision
        self.file_name = file_name
        self.device = device
        self._tokenizer = None
        self._model = None
        self.load_latency_seconds = None

    def load(self):
        """Load tokenizer and ONNX session, tracking startup latency."""
        if self._model is None:
            t0 = time.perf_counter()
            from transformers import AutoTokenizer
            from optimum.onnxruntime import ORTModelForCausalLM

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                revision=self.revision,
            )
            self._model = ORTModelForCausalLM.from_pretrained(
                self.model_id,
                subfolder="onnx",
                file_name=self.file_name,
                revision=self.revision,
            )
            self.load_latency_seconds = time.perf_counter() - t0
        return self

    def __call__(self, messages: list[dict[str, str]], profile: ModelProfile) -> str:
        if self._model is None:
            self.load()
        prompt_text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt_text, return_tensors="pt")
        max_tokens = min(profile.max_tokens, 256)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )
        prompt_len = inputs.input_ids.shape[1]
        gen_tokens = outputs[0][prompt_len:]
        return self._tokenizer.decode(gen_tokens, skip_special_tokens=True)


class HostedChatModelClient:
    """Hosted API client reading credentials exclusively from a configured environment variable.

    Supports `openai_compatible` (OpenAI, OpenRouter, custom base_url), `groq`
    (defaulting to `https://api.groq.com/openai/v1/chat/completions` and
    `GROQ_API_KEY`), and `gemini` REST APIs via stdlib `urllib.request`, never
    persisting or logging secret keys.
    """

    DEFAULT_BASE_URLS = {
        "openai_compatible": "https://api.openai.com/v1/chat/completions",
        "groq": "https://api.groq.com/openai/v1/chat/completions",
        "gemini": "https://generativelanguage.googleapis.com/v1beta/models",
    }
    DEFAULT_API_KEY_ENV_VARS = {
        "openai_compatible": "OPENAI_API_KEY",
        "groq": "GROQ_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }

    def __init__(
        self,
        *,
        provider: str,
        api_key_env_var: str | None = None,
        base_url: str | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
        min_call_interval_sec: float = 0.0,
    ):
        if provider not in self.DEFAULT_BASE_URLS:
            raise ValueError(f"Unsupported provider '{provider}'; expected one of {tuple(self.DEFAULT_BASE_URLS)}")
        if min_call_interval_sec < 0:
            raise ValueError("min_call_interval_sec must be nonnegative")
        self.provider = provider
        self.api_key_env_var = api_key_env_var or self.DEFAULT_API_KEY_ENV_VARS[provider]
        self.base_url = base_url or self.DEFAULT_BASE_URLS[provider]
        self.user_agent = user_agent
        self.min_call_interval_sec = float(min_call_interval_sec)
        self._last_call_monotonic: float | None = None

    def _pace_request(self) -> None:
        if self.min_call_interval_sec > 0.0:
            now = time.monotonic()
            if self._last_call_monotonic is not None:
                remaining = self.min_call_interval_sec - (now - self._last_call_monotonic)
                if remaining > 0.0:
                    time.sleep(remaining)
            self._last_call_monotonic = time.monotonic()

    def is_configured(self) -> bool:
        return bool(os.environ.get(self.api_key_env_var, "").strip())

    @staticmethod
    def _urlopen_json(req: urllib.request.Request, timeout_seconds: float) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            snippet = ""
            try:
                raw_err = exc.read(512)
                if raw_err:
                    snippet = raw_err.decode("utf-8", errors="replace").strip().replace("\n", " ")[:320]
            except Exception:
                snippet = ""
            if exc.code in NON_RETRYABLE_HTTP_STATUS_CODES:
                raise NonTransientHTTPError(exc.code, snippet) from exc
            hint = parse_retry_after_hint(getattr(exc, "headers", None), snippet) if exc.code == 429 else None
            raise TransientHTTPError(exc.code, snippet, retry_after_seconds=hint) from exc

    def __call__(self, messages: list[dict[str, str]], profile: ModelProfile) -> ModelClientResponse:
        validate_specialist_model_id(profile.model_id, allow_small_legacy=profile.allow_small_legacy)
        api_key = os.environ.get(self.api_key_env_var, "").strip()
        if not api_key:
            raise RuntimeError(f"Missing required API key environment variable: {self.api_key_env_var}")
        self._pace_request()

        if self.provider in ("openai_compatible", "groq"):
            body = json.dumps({
                "model": profile.model_id,
                "messages": messages,
                "temperature": profile.temperature,
                "max_tokens": profile.max_tokens,
            }).encode("utf-8")
            req = urllib.request.Request(
                self.base_url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": self.user_agent,
                },
                method="POST",
            )
            data = self._urlopen_json(req, profile.timeout_seconds)
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return ModelClientResponse(
                raw_text=text,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                provider_response_id=data.get("id"),
            )

        # Gemini generateContent REST API
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        contents = [
            {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [{"text": m["content"]}],
            }
            for m in messages
            if m["role"] != "system"
        ]
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": profile.temperature,
                "maxOutputTokens": profile.max_tokens,
            },
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        url = f"{self.base_url.rstrip('/')}/{profile.model_id}:generateContent?key={api_key}"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": self.user_agent,
            },
            method="POST",
        )
        data = self._urlopen_json(req, profile.timeout_seconds)
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        usage = data.get("usageMetadata", {})
        return ModelClientResponse(
            raw_text=text,
            prompt_tokens=usage.get("promptTokenCount"),
            completion_tokens=usage.get("candidatesTokenCount"),
            total_tokens=usage.get("totalTokenCount"),
            provider_response_id=data.get("responseId"),
        )


def build_phase4_role_backends(
    client: Callable[[list[dict[str, str]], ModelProfile], str | ModelClientResponse | dict[str, Any]] | None,
    *,
    provider: str,
    model_id: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
    timeout_seconds: float = 30.0,
    max_retries: int = 2,
    backoff_seconds: float = 0.25,
    prompt_version: str = "v1",
    budget: LiveCallBudget | None = None,
    min_call_interval_sec: float | None = None,
) -> dict[str, ModelAgentAdapter]:
    """Build four role-isolated specialist backends conforming to Phase 4 >=7B policy."""
    if min_call_interval_sec is not None and isinstance(client, HostedChatModelClient):
        client.min_call_interval_sec = float(min_call_interval_sec)
    return {
        role: ModelAgentAdapter(
            make_specialist_profile(
                role,
                provider=provider,
                model_id=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                backoff_seconds=backoff_seconds,
                prompt_version=prompt_version,
            ),
            client=client,
            budget=budget,
        )
        for role in ROLES
    }
