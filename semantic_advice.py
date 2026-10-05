"""Optional version-bound NLI annotations. This module never submits inference.

It reuses the schema-2 client contract and core audit journal. Import only when
client dependencies are installed; the core package does not import this adapter.
"""

from dataclasses import dataclass
import json
from typing import Callable
import uuid

from cert_recovery import CertificateSystem, Ref
from cert_recovery.crypto import canonical_json, sha256
from cert_recovery.models import Artifact
from remote_model_client import PROFILES, RESULT_SCHEMA_VERSION, normalize_output, redact


@dataclass(frozen=True)
class PreparedAdvice:
    """Immutable request data; input order is premise then hypothesis."""

    nonce: str
    profile: str
    bindings: tuple[tuple[Ref, str], ...]
    input_json: str

    @property
    def request_id(self) -> str:
        # Replacing bindings or text necessarily produces a different request ID.
        return "advice_" + sha256({"nonce": self.nonce, "profile": self.profile,
                                  "bindings": [{**ref.to_dict(), "content_hash": digest}
                                               for ref, digest in self.bindings],
                                  "input": json.loads(self.input_json)})

    def to_case(self) -> dict:
        # No gold NLI or dependency labels accompany artifact analysis.
        return {"id": self.request_id, "input": json.loads(self.input_json), "expected_label": None}


class SemanticAdviceAdapter:
    def __init__(self, system: CertificateSystem, text_of: Callable[[Artifact], str] | None = None):
        self.system = system
        self.text_of = text_of or (lambda node: canonical_json(node.payload))

    def prepare(self, premise: Ref, hypothesis: Ref, *, profile: str = "nli-zero-gpu") -> PreparedAdvice:
        if profile not in PROFILES:
            raise ValueError("Advice requires a reviewed profile")
        nodes = tuple(self.system.artifact(ref) for ref in (premise, hypothesis))
        if not all(self.system.is_valid(node.ref) for node in nodes):
            raise ValueError("Cannot prepare advice from stale or invalid artifacts")
        texts = tuple(self.text_of(node) for node in nodes)
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Artifact text projections must be nonempty strings")
        return PreparedAdvice(uuid.uuid4().hex, profile,
                              tuple((node.ref, node.content_hash) for node in nodes),
                              canonical_json({"premise": texts[0], "hypothesis": texts[1]}))

    def consume(self, prepared: PreparedAdvice, response: dict) -> dict:
        """Consume a returned schema-2 record, checking freshness on every call.

        A later consumer must call this again rather than treat an earlier audit
        annotation as perpetually fresh. No requests or automatic retries occur.
        """
        try:
            snapshot = json.loads(canonical_json(redact(response)))
        except (ValueError, TypeError):
            snapshot = {"unserializable_response_type": type(response).__name__}
        annotation = {"request_id": prepared.request_id, "profile": prepared.profile,
                      "input": json.loads(prepared.input_json), "disposition": "IGNORED",
                      "reason": "malformed_response_record", "nli_relation": None,
                      "normalized_output": None, "response": snapshot}
        try:
            reason, normalized = self._interpret(prepared, snapshot)
            annotation["reason"] = reason
            annotation["normalized_output"] = normalized
            if normalized is not None and normalized["branch"] == "model":
                annotation.update(disposition="ACCEPTED", nli_relation=normalized["label"])
        except (TypeError, ValueError, KeyError, RuntimeError) as error:
            annotation["validation_error"] = {"type": type(error).__name__, "message": redact(str(error))}
        return self.system.record_semantic_advice(prepared.bindings, annotation)

    @staticmethod
    def _interpret(prepared: PreparedAdvice, response: dict):
        if not isinstance(response, dict) or response.get("schema_version") != RESULT_SCHEMA_VERSION:
            raise ValueError("Expected a schema-2 response record")
        if (response.get("test_id") != prepared.request_id or response.get("run_id") != prepared.request_id
                or response.get("input") != json.loads(prepared.input_json)):
            return "response_request_mismatch", None
        profile = PROFILES[prepared.profile]
        if (response.get("profile") != prepared.profile or response.get("space_id") != profile.space_id
                or response.get("api_name") != profile.api_name
                or response.get("reviewed_space_revision") != profile.reviewed_space_revision):
            return "response_profile_mismatch", None
        # Do not promote configured IDs or an observed Hub revision to identity.
        if (response.get("verified_loaded_model_id") is not None
                or response.get("loaded_model_revision") is not None
                or response.get("model_identity_status") != "UNKNOWN"):
            return "unsupported_model_attestation", None
        if response.get("transport_status") != "SUCCESS":
            return "advice_unavailable", None
        if response.get("parsing_status") != "SUCCESS" or response.get("execution_status") != "SUCCESS":
            return "malformed_service_response", None
        # Reuse the reviewed parser instead of trusting a caller-modified label.
        normalized = normalize_output(prepared.profile, response["output"])
        classification = "SUCCESS" if normalized["branch"] == "model" else normalized["branch"].upper()
        if response.get("normalized_output") != normalized or response.get("classification_status") != classification:
            return "response_normalization_mismatch", None
        if normalized["branch"] == "heuristic":
            return "heuristic_output_is_not_model_advice", normalized
        if normalized["branch"] == "unknown":
            return "ambiguous_advice_branch", normalized
        return "model_nli_annotation_only; dependency_judgment_not_established", normalized
