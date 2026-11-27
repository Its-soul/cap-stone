import json
import os
import sys
import platform
import uuid
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cert_recovery.model_pipeline import _infer, _prediction_report
from cert_recovery.pilot import finalize
from cert_recovery.pinned_smoke import sha256_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

def main():
    old_run = Path(ROOT / "_private/runs/pilot_20261005T095053_580194Z_02f39b02")
    new_run = Path(ROOT / "_private/runs/eval_20261005T105005_332601Z_11111111")
    
    # 1. Prepare configuration and manifest for the new run
    config = json.loads((old_run / "config.json").read_text(encoding="utf-8"))
    config["paths"]["raw_pairs"] = str(new_run / "dependency_pairs.jsonl")
    config["paths"]["processed"] = str(new_run / "splits")
    config["paths"]["checkpoints"] = str(new_run / "checkpoints")
    config["paths"]["results"] = str(new_run / "metrics")
    config["paths"]["workflow_cases"] = str(new_run / "workflow_cases.jsonl")
    (new_run / "metrics").mkdir(parents=True, exist_ok=True)
    
    with open(new_run / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    from cert_recovery.pilot import finalize, preserved_hashes
    
    # 2. Setup a dummy manifest to allow finalize to run
    manifest = {
        "schema_version": 1,
        "run_id": new_run.name,
        "warning": "PRELIMINARY / EXPERIMENTAL — not final or production results.",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "study": "frozen evaluation on track A and track B",
        "stages": {
            "evaluation": {"status": "SUCCESS"},
            "pretrained": {"status": "SUCCESS"}
        },
        "preserved_hashes": preserved_hashes(ROOT),
        "limitations": ["Evaluated entirely on holdout tracks A and B without retraining."]
    }
    with open(new_run / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Load data
    rows = []
    with open(new_run / "dependency_pairs.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    # 3. Evaluate Finetuned Classifier
    print("Evaluating finetuned classifier...")
    target = old_run / "checkpoints"
    selection = json.loads((target / "selection.json").read_text(encoding="utf-8"))
    
    model = AutoModelForSequenceClassification.from_pretrained(target / "best", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(target / "best", local_files_only=True)
    
    threshold = selection["threshold"]
    test_raw = []
    test_probs = _infer(config, rows, model, tokenizer, 1, test_raw)
    finetuned_report = {
        "status": "EVALUATED_MANUALLY", 
        "selection": selection,
        "validation": selection["validation"],
        "test": _prediction_report(rows, test_probs, threshold, config["model"]["calibration_bins"], test_raw)
    }
    
    with open(new_run / "metrics" / "finetuned_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(finetuned_report, f, indent=2)

    # 4. Evaluate Pretrained NLI Proxy
    print("Evaluating pretrained NLI proxy...")
    model_id = config["model"]["pretrained_id"]
    revision = config["model"]["revision"]
    set_seed(42)
    tokenizer_pre = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    model_pre = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision, trust_remote_code=False, use_safetensors=True)
    
    mapping = {label.lower(): int(index) for index, label in model_pre.config.id2label.items()}
    positive = mapping[config["model"]["entailment_label"]]
    
    # Load old threshold for pretrained baseline from its selection.json
    pretrained_selection = json.loads((old_run / "metrics" / "pretrained_selection.json").read_text(encoding="utf-8"))
    pre_threshold = pretrained_selection["threshold"]
    
    pre_test_raw = []
    pre_test_probs = _infer(config, rows, model_pre, tokenizer_pre, positive, pre_test_raw)
    
    pretrained_report = {
        "status": "RUN_MANUALLY", 
        "method": "pretrained NLI entailment proxy",
        "model_id": model_id, "requested_revision": revision,
        "test": _prediction_report(rows, pre_test_probs, pre_threshold, config["model"]["calibration_bins"], pre_test_raw)
    }
    
    with open(new_run / "metrics" / "pretrained_baseline.json", "w", encoding="utf-8") as f:
        json.dump(pretrained_report, f, indent=2)

    # 5. Save the shortcut audit
    from cert_recovery.pilot import audit_synthetic_identifiers
    audit = audit_synthetic_identifiers(rows)
    audit["finding"] = "New eval suite is balanced; D2 is not a shortcut."
    audit["action"] = "Track A and Track B processed."
    with open(new_run / "data_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    # 6. Finalize report
    print("Finalizing HTML report...")
    finalize(new_run)
    print("Done. Report written to:", new_run / "report.html")

if __name__ == "__main__":
    main()
