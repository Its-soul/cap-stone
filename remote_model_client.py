"""Remote-only clients for reviewed public NLI Spaces; no model imports/downloads."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
import math
import re
from time import perf_counter
import uuid

from gradio_client import Client
from huggingface_hub import HfApi


PROJECT_MODEL = "cross-encoder/nli-deberta-v3-small"
RESULT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class SpaceProfile:
    space_id: str
    model_id: str  # Reviewed source configuration/fallback, never loaded-weight attestation.
    api_name: str
    reviewed_space_revision: str
    parameter_names: tuple[str, ...]
    output_components: tuple[str, ...]
    zero_gpu: bool
    pipeline: str


# Names, parameter order, outputs, and revisions were read from live API schemas
# and Space source. Changes require another source/schema review, not guesses.
PROFILES = {
    "nli-zero-gpu": SpaceProfile(
        "MridulSharma02/contradiction-detector",
        "cross-encoder/nli-MiniLM2-L6-H768",
        "/contradiction_detector",
        "f08cb1f21f447e96f7703b9c6b7c477ca2843f91",
        ("sentence_a", "sentence_b"), ("Html",), True,
        "NLTK disconnection heuristic, then CrossEncoder with unverified local/fallback weights; rounded HTML output",
    ),
    "nli-cpu": SpaceProfile(
        "tasksource/ModernBERT-zero-shot-nli",
        "tasksource/ModernBERT-base-nli",
        "/process_input",
        "7660f8089995f96ee3d4d5a36bae734637275463",
        ("text_input", "labels_or_premise", "mode"), ("Label", "Html"), False,
        "Transformers paired text classification in Natural Language Inference mode",
    ),
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def redact(value, token=None):
    """Remove credentials and encode non-finite numbers for strict JSON logging."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_number": str(value)}
    if isinstance(value, str):
        if token:
            value = value.replace(token, "[REDACTED]")
        return re.sub(r"hf_[A-Za-z0-9]{20,}", "[REDACTED]", value)
    if isinstance(value, dict):
        return {key: redact(item, token) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item, token) for item in value]
    return value


class RemoteModelError(RuntimeError):
    pass


class ResponseValidationError(RemoteModelError):
    """An invalid service response, with only observations made before failure."""

    def __init__(self, code, message, observations=None):
        super().__init__(message)
        self.code = code
        self.observations = observations or {}


class _Text(HTMLParser):
    """Validate the reviewed HTML fragment, rather than repairing broken HTML."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.stack = []
        self.has_div = False

    def handle_starttag(self, tag, attrs):
        if tag not in {"div", "strong", "b", "br", "hr"}:
            raise ResponseValidationError("malformed_html", f"Unexpected HTML tag: {tag}")
        if tag == "div":
            self.has_div = True
        if tag not in {"br", "hr"}:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in {"br", "hr"}:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if not self.stack or self.stack.pop() != tag:
            raise ResponseValidationError("malformed_html", "Unbalanced HTML tags")

    def handle_data(self, data):
        if "<" in data or ">" in data:
            raise ResponseValidationError("malformed_html", "Malformed HTML text")
        self.parts.append(data)


def _probability(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RemoteModelError("Remote confidence is not numeric")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise RemoteModelError("Remote confidence is outside [0, 1]")
    return float(value)


def normalize_output(profile_name, raw):
    """Interpret only a reviewed profile; branch/origin evidence is inferred.

    No remote response is certificate authority. A valid service response may
    have no canonical NLI label when it came from a heuristic or unknown branch.
    """
    if profile_name == "nli-zero-gpu":
        return _normalize_wrapped_nli(raw)
    if profile_name == "nli-cpu":
        return _normalize_cpu_nli(raw)
    raise ResponseValidationError("unknown_profile", "No reviewed response contract for this profile")


def _normalize_wrapped_nli(raw):
    if not isinstance(raw, str):
        raise ResponseValidationError("malformed_response", "Expected HTML text from the reviewed endpoint")
    if "Please enter both sentences." in raw:
        raise ResponseValidationError("empty_input_warning", "Service returned an empty-input warning")
    if re.search(r"quota|GPU.*(?:limit|exceeded)", raw, re.IGNORECASE) and "Confidence:" not in raw:
        raise ResponseValidationError("quota_exceeded", "Service returned a quota error")
    parser = _Text()
    parser.feed(raw)
    parser.close()
    if parser.stack or not parser.has_div:
        raise ResponseValidationError("malformed_html", "Expected a complete reviewed HTML fragment")
    parts = [part.strip() for part in parser.parts if part.strip()]
    if not parts:
        raise ResponseValidationError("missing_label", "Response has no service label")
    # The heading is the first visible text, not an arbitrary token in NLTK data.
    original_label = parts[0]
    for emoji in ("✅", "❌", "↔️"):
        if original_label.startswith(emoji):
            original_label = original_label[len(emoji):].strip()
            break
    observed = {"original_label": original_label}
    mapping = {"Consistent": "entailment", "Contradiction": "contradiction", "Unrelated": "neutral"}
    if original_label.startswith("Confidence:"):
        raise ResponseValidationError("missing_label", "Response has confidence but no service label")
    if original_label not in mapping:
        raise ResponseValidationError("unknown_label", f"Unknown service label: {original_label}", observed)
    if any(part in mapping for part in parts[1:]):
        raise ResponseValidationError("malformed_response", "Response contains multiple service labels", observed)
    text = " ".join(parts)
    confidence_fields = re.findall(r"Confidence:\s*(\S+)", text)
    if len(confidence_fields) != 1:
        raise ResponseValidationError("missing_confidence", "Expected exactly one displayed confidence", observed)
    confidence_text = confidence_fields[0]
    observed["confidence_text"] = confidence_text
    try:
        if not confidence_text.endswith("%"):
            raise ValueError("Confidence must be a percentage")
        confidence = _probability(float(confidence_text[:-1]) / 100)
    except (ValueError, RemoteModelError) as error:
        raise ResponseValidationError("invalid_confidence", str(error), observed) from error
    features = {}
    for heading, name in (("Token Overlap", "token_overlap"), ("WordNet Similarity", "wordnet_similarity")):
        values = re.findall(re.escape(heading) + r":\s*(\S+)", text)
        if not values:
            features[name] = None
            continue
        try:
            if len(values) != 1:
                raise ValueError("Duplicate NLTK feature")
            features[name] = _probability(float(values[0]))
        except (ValueError, RemoteModelError) as error:
            raise ResponseValidationError("invalid_nltk_features", str(error), observed) from error
    known_features = all(value is not None for value in features.values())
    # The pinned preprocessor rounds these features BEFORE applying the filter;
    # the HTML returns those same values. This is inference from source, not a
    # runtime branch attestation or proof that any particular weights executed.
    filtered = known_features and features["token_overlap"] == 0.0 and features["wordnet_similarity"] < 0.12
    if filtered and original_label != "Unrelated":
        raise ResponseValidationError("inconsistent_branch", "Label conflicts with the reviewed NLTK filter", observed)
    if original_label != "Unrelated":
        branch, evidence = "model", "inferred_from_service_label_and_reviewed_source"
    elif known_features:
        branch = "heuristic" if filtered else "model"
        evidence = "inferred_from_returned_nltk_features_and_reviewed_source"
    else:
        branch, evidence = "unknown", "unknown"
    origin = {"model": "rounded_model_softmax", "heuristic": "heuristic_score", "unknown": "unknown"}[branch]
    return {"original_label": original_label, "label": mapping[original_label] if branch == "model" else None,
            "branch": branch, "branch_evidence": evidence, "branch_attested": False,
            "confidence_text": confidence_text, "confidence": confidence, "confidence_origin": origin,
            "probabilities": None, "nltk_features": features, "advisory_only": True}


def _normalize_cpu_nli(raw):
    if not isinstance(raw, (tuple, list)) or len(raw) != 2 or not isinstance(raw[0], dict) or not isinstance(raw[1], str):
        raise ResponseValidationError("malformed_response", "Expected a Label object and HTML from the reviewed endpoint")
    original_label = raw[0].get("label")
    observed = {"original_label": original_label} if isinstance(original_label, str) else {}
    if not isinstance(original_label, str) or not original_label:
        raise ResponseValidationError("missing_label", "Response has no service label")
    if original_label.lower() not in {"contradiction", "entailment", "neutral"}:
        raise ResponseValidationError("unknown_label", f"Unknown service label: {original_label}", observed)
    scores = raw[0].get("confidences")
    if not isinstance(scores, list) or len(scores) != 3:
        raise ResponseValidationError("missing_confidence", "Expected three NLI probabilities", observed)
    probabilities = {}
    for row in scores:
        if not isinstance(row, dict) or not isinstance(row.get("label"), str) or "confidence" not in row:
            raise ResponseValidationError("malformed_response", "Malformed NLI probability row", observed)
        name = row["label"].lower()
        if name in probabilities:
            raise ResponseValidationError("unknown_label", "Duplicate NLI probability label", observed)
        try:
            probabilities[name] = _probability(row["confidence"])
        except RemoteModelError as error:
            raise ResponseValidationError("invalid_confidence", str(error), observed) from error
    if set(probabilities) != {"contradiction", "entailment", "neutral"}:
        raise ResponseValidationError("unknown_label", "Unexpected NLI probability labels", observed)
    if not math.isclose(sum(probabilities.values()), 1, abs_tol=0.001):
        raise ResponseValidationError("invalid_confidence", "NLI probabilities do not sum to one", observed)
    label = max(probabilities, key=probabilities.get)
    if original_label.lower() != label:
        raise ResponseValidationError("inconsistent_label", "Top label disagrees with returned probabilities", observed)
    return {"original_label": original_label, "label": label, "branch": "model",
            "branch_evidence": "inferred_from_reviewed_source_contract", "branch_attested": False,
            "confidence_text": None, "confidence": probabilities[label], "confidence_origin": "model_softmax",
            "probabilities": probabilities, "advisory_only": True}


def create_case_record(profile_name, case, run_id):
    """Schema 2 uses existing test_id/output/api_name keys; output is the raw response."""
    profile = PROFILES[profile_name]
    return {"schema_version": RESULT_SCHEMA_VERSION, "run_id": run_id or f"request_{uuid.uuid4().hex}", "test_id": case["id"],
            "timestamp": utc_now(), "input": dict(case["input"]), "expected_label": case["expected_label"],
            "space_id": profile.space_id, "api_name": profile.api_name, "profile": profile_name,
            "reviewed_space_revision": profile.reviewed_space_revision,
            "configured_model_id": PROJECT_MODEL, "fallback_model_id": profile.model_id,
            "verified_loaded_model_id": None, "loaded_model_revision": None,
            "model_identity_status": "UNKNOWN", "output": None, "normalized_output": None,
            "execution_status": "NOT_RUN", "transport_status": "NOT_RUN", "parsing_status": "NOT_RUN",
            "classification_status": "NOT_RUN", "matches_expected_label": None, "error": None,
            "latency_seconds": None,
            "latency_scope": "end-to-end client request, including network, queue, service and response validation"}


class RemoteModelClient:
    def __init__(self, profile_name="nli-zero-gpu", *, token=None, timeout=120):
        if profile_name not in PROFILES:
            raise RemoteModelError("No reviewed Space profile with this name")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Timeout must be finite and positive")
        self.profile_name = profile_name
        self.profile = PROFILES[profile_name]
        self.token = token or None
        self.timeout = timeout
        self.api = HfApi(token=self.token or False)
        self.client = None
        profile_metadata = asdict(self.profile)
        profile_metadata.pop("model_id")
        self.metadata = {**profile_metadata, "profile": profile_name,
                         "configured_model_id": PROJECT_MODEL, "fallback_model_id": self.profile.model_id,
                         "source_model_candidates": (["fine-tuned-model", self.profile.model_id]
                                                     if self.profile.zero_gpu else [self.profile.model_id]),
                         "is_project_model": False, "verified_loaded_model_id": None,
                         "model_identity_status": "UNKNOWN",
                         "model_identity_source": "reviewed source candidates only; loaded weights not attested",
                         "loaded_model_revision": None, "authenticated": bool(self.token)}

    def connect(self, *, allow_alternative_model=False):
        if not allow_alternative_model:
            raise RemoteModelError(
                f"This reviewed Space configures {self.profile.model_id}; its loaded weights are unverified. "
                "Pass --allow-alternative-model to run a separate NLI baseline. "
                "No working compatible endpoint for the project checkpoint was found."
            )
        info = self.api.space_info(self.profile.space_id, timeout=min(self.timeout, 30))
        runtime = info.runtime
        self.metadata.update(space_revision=info.sha, space_host=info.host,
                             runtime_revision=runtime.raw.get("sha") if runtime else None,
                             stage=str(runtime.stage) if runtime else None,
                             hardware=str(runtime.hardware) if runtime and runtime.hardware else None)
        if not runtime or str(runtime.stage) != "RUNNING":
            raise RemoteModelError(f"Space is {self.metadata['stage']}; no inference submitted")
        if info.sha != self.profile.reviewed_space_revision or (
            runtime.raw.get("sha") and runtime.raw["sha"] != info.sha
        ):
            raise RemoteModelError("Space revision changed or deployment differs; review its source and API before updating the profile")
        hardware = str(runtime.hardware or "")
        if self.profile.zero_gpu and not hardware.startswith("zero-"):
            raise RemoteModelError(f"Expected ZeroGPU, found {hardware}; no inference submitted")
        if not self.profile.zero_gpu and hardware != "cpu-basic":
            raise RemoteModelError(f"Expected free CPU hardware, found {hardware}; no inference submitted")
        if info.sdk != "gradio":
            raise RemoteModelError("Space no longer uses the reviewed Gradio SDK")
        self.client = Client(self.profile.space_id, token=self.token or False, verbose=False,
                             max_workers=1, download_files=False, analytics_enabled=False,
                             httpx_kwargs={"timeout": min(self.timeout, 30)})
        schema = self.client.view_api(return_format="dict", print_info=False)
        endpoint = schema.get("named_endpoints", {}).get(self.profile.api_name)
        if not endpoint:
            raise RemoteModelError("Reviewed endpoint is absent from the live API schema")
        parameters = endpoint.get("parameters", [])
        if tuple(p.get("parameter_name") for p in parameters) != self.profile.parameter_names:
            raise RemoteModelError("Endpoint parameters changed; no inference submitted")
        if any(p.get("type", {}).get("type") != "string" for p in parameters):
            raise RemoteModelError("Endpoint input types changed; no inference submitted")
        if tuple(r.get("component") for r in endpoint.get("returns", [])) != self.profile.output_components:
            raise RemoteModelError("Endpoint outputs changed; no inference submitted")
        if self.profile_name == "nli-cpu" and "Natural Language Inference" not in parameters[2]["type"].get("enum", []):
            raise RemoteModelError("Required NLI mode is absent from the API schema")
        dependency = next((d for d in self.client.config.get("dependencies", [])
                           if d.get("api_name") == self.profile.api_name.lstrip("/")), None)
        if self.profile.zero_gpu and (not dependency or dependency.get("zerogpu") is not True):
            raise RemoteModelError("Endpoint is not advertised as ZeroGPU; no inference submitted")
        self.metadata.update(endpoint_schema=endpoint, gradio_version=self.client.config.get("version"),
                             endpoint_zero_gpu=bool(dependency and dependency.get("zerogpu")))
        # This is the current Hub revision, not proof of which weights a shared
        # running Space has cached. Failure to read it does not change inference.
        try:
            self.metadata["model_hub_revision_observed"] = self.api.model_info(self.profile.model_id, timeout=15).sha
        except Exception as error:
            self.metadata["model_metadata_error"] = redact(str(error), self.token)
        return self.metadata

    def quota(self):
        if not self.profile.zero_gpu:
            return {"status": "NOT_APPLICABLE", "remaining_gpu_seconds": None}
        if not self.token:
            return {"status": "UNKNOWN", "remaining_gpu_seconds": None,
                    "reason": "Anonymous quota is not available through the authenticated account quota API"}
        try:
            quota = self.api.get_zero_gpu_quota()
            return {"status": "AVAILABLE", "base_gpu_seconds": quota.base,
                    "remaining_gpu_seconds": quota.remaining,
                    "resets_at": quota.resets_at.isoformat() if quota.resets_at else None,
                    "overquota_used_gpu_seconds": quota.overquota_used}
        except Exception as error:
            return {"status": "UNKNOWN", "remaining_gpu_seconds": None,
                    "error": redact(str(error), self.token)}

    def infer(self, case, *, run_id=None):
        record = create_case_record(self.profile_name, case, run_id)
        record["execution_status"] = "ERROR"
        start = perf_counter()
        job = None
        stage = "preflight"
        try:
            if self.client is None:
                raise RemoteModelError("Connect and validate the API before inference")
            pair = case["input"]
            if self.profile_name == "nli-zero-gpu":
                arguments = {"sentence_a": pair["premise"], "sentence_b": pair["hypothesis"]}
            else:
                arguments = {"text_input": pair["premise"], "labels_or_premise": pair["hypothesis"],
                             "mode": "Natural Language Inference"}
            stage = "transport"
            job = self.client.submit(api_name=self.profile.api_name, **arguments)
            record["output"] = job.result(timeout=self.timeout)
            record["transport_status"] = "SUCCESS"
            stage = "response_validation"
            record["normalized_output"] = normalize_output(self.profile_name, record["output"])
            record["parsing_status"] = "SUCCESS"
            normalized = record["normalized_output"]
            if normalized["label"] is not None:
                record["classification_status"] = "SUCCESS"
                if case["expected_label"] is not None:
                    record["matches_expected_label"] = normalized["label"] == case["expected_label"]
            else:
                record["classification_status"] = normalized["branch"].upper()
            record["execution_status"] = "SUCCESS"
        except TimeoutError as error:
            record["cancellation_requested"] = job is not None
            if job is not None:
                try:
                    job.cancel()
                except Exception as cancellation_error:
                    record["cancellation_error"] = redact(str(cancellation_error), self.token)
            record["execution_status"] = "TIMEOUT"
            record["transport_status"] = "TIMEOUT"
            record["error"] = {"type": type(error).__name__,
                               "stage": stage, "code": "timeout",
                               "message": (f"No result within {self.timeout}s; "
                                           + ("cancellation requested" if job is not None else "no job handle available for cancellation")
                                           + ", remote outcome unknown")}
        except Exception as error:
            if stage == "response_validation":
                record["parsing_status"] = "ERROR"
                record["classification_status"] = "ERROR"
            elif stage == "transport":
                record["transport_status"] = "ERROR"
            message = redact(str(error), self.token)
            code = getattr(error, "code", "remote_error")
            if stage == "transport":
                if re.search(r"quota|GPU.*(?:limit|exceeded)", message, re.IGNORECASE):
                    code = "quota_exceeded"
                elif re.search(r"unavailable|paused|sleeping|503", message, re.IGNORECASE):
                    code = "space_unavailable"
            record["error"] = {"type": type(error).__name__, "stage": stage, "code": code, "message": message}
            if isinstance(error, ResponseValidationError) and error.observations:
                record["error"]["observations"] = error.observations
        finally:
            record["latency_seconds"] = perf_counter() - start
            record["completed_at"] = utc_now()
        return redact(record, self.token)

    def close(self):
        if self.client is not None:
            self.client.close()
