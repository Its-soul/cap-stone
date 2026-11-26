"""Read-only inference using the frozen candidate V2 checkpoint and threshold."""

import json
import os
from pathlib import Path
from threading import Lock

from .crypto import sha256
from .model_pipeline import _infer
from .pinned_smoke import sha256_file


LABELS = {0: "independent", 1: "depends_on"}


def read_record(path):
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"{path.name}: {error}") from error
    if not isinstance(record, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return record


def validate_checkpoint(run):
    """Validate inference artifacts independently of the historical training replay."""
    manifest = read_record(run / "manifest.json")
    protocol = read_record(run / "protocol.json")
    config = read_record(run / "config.json")
    selection = read_record(run / "checkpoints" / "selection.json")
    frozen = read_record(run / "comparison_selection_frozen.json")
    if manifest.get("status") != "SUCCESS" or manifest.get("protocol_sha256") != sha256(protocol):
        raise ValueError("V2 run is incomplete or its frozen protocol hash differs")
    if sha256(config) != protocol.get("config_sha256") or selection.get("config_hash") != protocol.get("config_sha256"):
        raise ValueError("V2 configuration hash differs from the frozen run or selection")
    if sha256(selection) != frozen.get("candidate_selection_sha256"):
        raise ValueError("V2 checkpoint selection or its saved threshold differs from the frozen record")
    threshold = selection.get("threshold")
    if type(threshold) not in (int, float) or not 0 <= threshold <= 1:
        raise ValueError("V2 selection has an invalid saved threshold")

    checkpoint = run / "checkpoints" / "best"
    hashes = selection.get("checkpoint_file_hashes")
    required = {"model.safetensors", "config.json", "tokenizer.json"}
    if (not isinstance(hashes, dict) or not required <= hashes.keys()
            or hashes["model.safetensors"] != selection.get("checkpoint_sha256")):
        raise ValueError("V2 selection is missing the recorded model or tokenizer hashes")
    filenames = {path.name for path in checkpoint.iterdir() if path.is_file()}
    if missing := sorted(hashes.keys() - filenames):
        raise ValueError(f"Missing V2 checkpoint file: {missing[0]}")
    if filenames - hashes.keys():
        raise ValueError("V2 checkpoint contains files absent from the frozen selection")
    for filename, digest in hashes.items():
        if sha256_file(checkpoint / filename) != digest:
            raise ValueError(f"V2 checkpoint hash mismatch: {filename}")
    return config, checkpoint, threshold


class V2Predictor:
    def __init__(self, root):
        self.runs = Path(root) / "_private" / "runs"
        self.lock = Lock()
        self.loaded = None

    def load(self):
        configured = os.environ.get("CERT_RECOVERY_V2_RUN")
        if configured:
            run = Path(configured).expanduser().resolve()
        else:
            runs = [path for path in self.runs.glob("candidate_v2_*") if path.is_dir()]
            if not runs:
                raise FileNotFoundError("No V2 run bundle found. Set CERT_RECOVERY_V2_RUN to its directory.")
            if len(runs) > 1:
                raise ValueError("Multiple V2 run bundles found. Set CERT_RECOVERY_V2_RUN to the one to use.")
            run = runs[0]
        config, checkpoint, threshold = validate_checkpoint(run)

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
        model, loading = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False, use_safetensors=True, output_loading_info=True
        )
        if model.config.id2label != LABELS:
            raise ValueError("V2 checkpoint labels must be independent and depends_on")
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("V2 model parameters did not load exactly as saved")
        return config, threshold, model, tokenizer

    def predict(self, source, assessment):
        if any(not isinstance(value, str) or not value.strip() or len(value) > 4000
               for value in (source, assessment)):
            raise ValueError("Enter two pieces of text, each between 1 and 4,000 characters")

        with self.lock:
            try:
                if self.loaded is None:
                    self.loaded = self.load()
                config, threshold, model, tokenizer = self.loaded
                capture = []
                probability, = _infer(config, [{"premise": source, "hypothesis": assessment}],
                                      model, tokenizer, 1, capture)
                scores = capture[0]["original_model_class_probabilities"]
                if any(not 0 <= score <= 1 for score in scores.values()):
                    raise ValueError("V2 returned invalid model scores")
            except Exception as error:
                return {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"}

        return {"status": "ok", "prediction": LABELS[int(probability >= threshold)],
                "scores": scores, "threshold": threshold}
