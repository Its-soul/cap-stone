"""Optional direct-model Colab smoke; no imports/downloads before manual guards.

This records one fixed pair per model, without a Space heuristic wrapper, research
metrics, certificate authority, or automatic retries. Offline tests use fixtures.
"""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
from time import perf_counter
import uuid

from .config import require_manual


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_label_order(config, reviewed_order):
    labels = {int(index): label.lower() for index, label in config["id2label"].items()}
    if set(labels) != {0, 1, 2}:
        raise ValueError("Expected exactly three label indices")
    order = [labels[index] for index in range(3)]
    if (order != reviewed_order
            or set(order) != {"contradiction", "entailment", "neutral"}
            or config["label2id"] != {label: index for index, label in enumerate(order)}):
        raise ValueError("Configuration disagrees with reviewed three-class label order")
    return order


def prepare_smoke(root, profile):
    """Pure offline preflight; preserves directional text and checks immutable pins."""
    root = Path(root)
    spec = json.loads((root / "config/pinned_smoke_models.json").read_text(encoding="utf-8"))
    model = spec["models"][profile]
    revision = model["revision"]
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("An immutable 40-character model revision is required")
    validate_label_order(model["reviewed_config"], model["reviewed_label_order"])
    dataset_path = root / "baseline_tests.json"
    if sha256_file(dataset_path) != spec["dataset_sha256"]:
        raise ValueError("Fixed dataset hash changed; review separately before inference")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    # Deliberately bounded smoke, not the six-case baseline or an evaluation.
    if spec["case_ids"] != ["nli-003"]:
        raise ValueError("This prepared smoke is bounded to case nli-003")
    cases = [case for case in dataset["cases"] if case["id"] in spec["case_ids"]]
    if len(cases) != 1:
        raise ValueError("Missing or duplicate smoke case")
    return {"model": model, "cases": cases, "dataset_sha256": spec["dataset_sha256"],
            "spec_sha256": sha256_file(root / "config/pinned_smoke_models.json")}


def prediction_from_logits(logits, label_order):
    if len(logits) != 3 or any(not math.isfinite(value) for value in logits):
        raise ValueError("Expected three finite raw logits")
    # Explicit float64 softmax of captured logits; no missing probabilities or calibration claim.
    exponentials = [math.exp(value - max(logits)) for value in logits]
    probabilities = [value / sum(exponentials) for value in exponentials]
    index = max(range(3), key=lambda item: logits[item])
    return {"raw_logits": logits, "class_probabilities": dict(zip(label_order, probabilities)),
            "predicted_class_index": index, "canonical_prediction": label_order[index],
            "probability_computation": "stable_float64_softmax_of_captured_logits",
            "confidence_origin": "direct_model_softmax", "advisory_only": True}


def save_new(path, document):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def run_pinned_smoke(config, profile, *, manual=False):
    require_manual(config, "allow_model_inference", manual)
    root = Path(config["project_root"])
    plan = prepare_smoke(root, profile)
    # No model packages/network are touched until both guards and offline preflight pass.
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    versions = {name: version(name) for name in
                ("torch", "transformers", "huggingface-hub", "tokenizers", "safetensors", "sentencepiece",
                 "numpy", "accelerate", "datasets", "protobuf", "PyYAML")}
    if versions["transformers"] != "4.57.1":
        raise ValueError("Use the prepared requirements-colab.txt Transformers version")
    spec = plan["model"]
    model_id, revision = spec["model_id"], spec["revision"]
    config_path = Path(hf_hub_download(model_id, "config.json", revision=revision))
    if sha256_file(config_path) != spec["config_sha256"]:
        raise ValueError("Downloaded pinned model configuration differs from reviewed bytes")
    downloaded_config = json.loads(config_path.read_text(encoding="utf-8"))
    order = validate_label_order(downloaded_config, spec["reviewed_label_order"])
    actual_config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    if actual_config._commit_hash != revision:
        raise ValueError("Resolved configuration revision is not the reviewed immutable pin")
    validate_label_order(actual_config.to_dict(), order)
    run_id = "pinned_smoke_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + uuid.uuid4().hex[:8]
    destination = root / "results"
    destination.mkdir(parents=True, exist_ok=True)
    prepared = {"schema_version": 1, "run_id": run_id, "execution_status": "PREPARED",
                "workflow": "direct_pinned_model_smoke_without_space_wrapper", "profile": profile,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(), **plan,
                "downloaded_model_config": downloaded_config,
                "resolved_config_revision": actual_config._commit_hash,
                "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                            "packages": versions, "device": "cuda" if torch.cuda.is_available() else "cpu",
                            "cuda_version": torch.version.cuda},
                "tokenization": {"max_length": config["model"]["max_length"], "truncation": True,
                                 "padding": True, "input_order": "premise_then_hypothesis"},
                "code_sha256": sha256_file(Path(__file__)),
                "config_sha256": sha256_file(root / "config/config.yaml"), "seed": 42}
    # Persist inputs/configuration/runtime BEFORE any weight loading or inference.
    save_new(destination / (run_id + ".prepared.json"), prepared)
    result = {**prepared, "execution_status": "ERROR", "predictions": [], "error": None}
    start = perf_counter()
    try:
        torch.manual_seed(42)
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
        model, loading = AutoModelForSequenceClassification.from_pretrained(
            model_id, revision=revision, config=actual_config, trust_remote_code=False,
            use_safetensors=True, output_loading_info=True)
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("Checkpoint did not load exactly; refusing inference with altered parameters")
        if model.config._commit_hash != revision:
            raise ValueError("Loaded model configuration revision is unverified")
        validate_label_order(model.config.to_dict(), order)
        weight_path = Path(hf_hub_download(model_id, "model.safetensors", revision=revision, local_files_only=True))
        result["loaded_weights"] = {"model_id": model_id, "revision": revision,
                                    "filename": "model.safetensors", "sha256": sha256_file(weight_path),
                                    "evidence": "direct immutable loader, strict loading info and cached weight digest"}
        result["loading_info"] = loading
        model.to(prepared["runtime"]["device"]).eval()
        for case in plan["cases"]:
            pair = case["input"]
            features = tokenizer(pair["premise"], pair["hypothesis"], return_tensors="pt",
                                 max_length=prepared["tokenization"]["max_length"], padding=True, truncation=True)
            features = {key: value.to(prepared["runtime"]["device"]) for key, value in features.items()}
            with torch.no_grad():
                logits = model(**features).logits.detach().cpu().tolist()[0]
            prediction = prediction_from_logits(logits, order)
            result["predictions"].append({"case": case, **prediction,
                                          "matches_expected_label": prediction["canonical_prediction"] == case["expected_label"]})
        result["execution_status"] = "SUCCESS"
    except Exception as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        result["elapsed_seconds_including_loading"] = perf_counter() - start
        result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_new(destination / (run_id + ".json"), result)
    return result
