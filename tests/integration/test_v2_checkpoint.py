"""Opt-in, read-only replay of two V2 validation predictions on private weights."""
import json
import os
from pathlib import Path

import pytest

from cert_recovery.controlled_study import verify_frozen
from cert_recovery.data_pipeline import load_verified_splits
from cert_recovery.model_pipeline import _infer, _prediction_report
from cert_recovery.pinned_smoke import sha256_file


@pytest.mark.v2_checkpoint
def test_private_v2_checkpoint_reproduces_saved_validation_predictions(monkeypatch):
    location = os.environ.get("CERT_RECOVERY_V2_RUN")
    if not location:
        pytest.skip("Set CERT_RECOVERY_V2_RUN to opt in with an existing private V2 bundle.")
    run = Path(location).expanduser().resolve()
    config, _ = verify_frozen(run)
    selection = json.loads((run / "checkpoints/selection.json").read_text())
    checkpoint = run / "checkpoints/best"
    assert sha256_file(checkpoint / "model.safetensors") == selection["checkpoint_sha256"]
    for filename, digest in selection["checkpoint_file_hashes"].items():
        assert sha256_file(checkpoint / filename) == digest

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    torch = pytest.importorskip("torch", reason="Use the existing private model environment.")
    transformers = pytest.importorskip("transformers", reason="Use the existing private model environment.")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False, use_safetensors=True)
    assert model.config.id2label == {0: "independent", 1: "depends_on"}

    splits, _ = load_verified_splits(config)
    rows = splits["validation"][:2]
    assert len(rows) == 2
    for row in rows:
        assert len(tokenizer(row["premise"], row["hypothesis"], truncation=False)["input_ids"]) <= config["model"]["max_length"]
    saved = json.loads((run / "comparisons/validation_candidate_v2.json").read_text())
    expected = {row["pair_id"]: row for row in saved["predictions"]}
    capture = []
    probabilities = _infer(config, rows, model, tokenizer, 1, capture)
    report = _prediction_report(rows, probabilities, selection["threshold"], config["model"]["calibration_bins"], capture)
    for row, actual in zip(rows, report["predictions"]):
        recorded = expected[row["pair_id"]]
        assert all(recorded[key] == row[key] for key in ("premise", "hypothesis", "label"))
        assert actual["probability"] == pytest.approx(recorded["probability"], abs=1e-5, rel=1e-5)
        assert actual["prediction"] == recorded["prediction"]
        assert sum(actual["original_model_class_probabilities"].values()) == pytest.approx(1.0)
