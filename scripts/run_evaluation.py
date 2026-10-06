import json
import sys
import argparse
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cert_recovery.model_pipeline import _infer, _prediction_report
from cert_recovery.pilot import preserved_hashes
from cert_recovery.crypto import sha256
from cert_recovery.pinned_smoke import sha256_file

def compute_paired_metrics(rows, probs, threshold, pairs=None):
    from cert_recovery.paired_metrics import paired_metrics
    if pairs is None:
        return {"status": "UNAVAILABLE: explicit transformation manifest required"}
    if len(rows) != len(probs):
        raise ValueError("Every probability must have an input row")
    predictions = [{**row, "probability": p, "prediction": int(p >= threshold)} for row,p in zip(rows,probs)]
    return paired_metrics(rows, predictions, pairs)


def main():
    parser = argparse.ArgumentParser(description="Evaluate frozen checkpoint on new dataset")
    parser.add_argument("--checkpoint-run", required=True, help="Path to the pilot run containing the checkpoint")
    parser.add_argument("--eval-dataset", required=True, help="Path to the JSONL evaluation dataset")
    parser.add_argument("--output-dir", required=True, help="Path to write the new evaluation run")
    parser.add_argument("--pair-manifest", help="Explicit validated comparison/member IDs and transformations")
    args = parser.parse_args()

    old_run = Path(args.checkpoint_run).resolve()
    new_run = Path(args.output_dir).resolve()
    eval_dataset = Path(args.eval_dataset).resolve()

    if new_run.exists():
        if (new_run / "manifest.json").exists() and json.loads((new_run / "manifest.json").read_text()).get("execution_status") == "SUCCESS":
            raise FileExistsError("Success result already exists; will not overwrite.")
        raise FileExistsError("Attempt directory already exists; preserve failed/pending evidence and use a new directory")
    new_run.mkdir(parents=True, exist_ok=False)

    # Start pending state
    device = "not loaded"
    manifest = {
        "schema_version": 1,
        "run_id": new_run.name,
        "warning": "PRELIMINARY / EXPERIMENTAL - not final or production results.",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "execution_status": "PENDING",
        "study": "frozen evaluation",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "device": device
        },
        "stages": {
            "evaluation": {"status": "PENDING"},
            "pretrained": {"status": "PENDING"},
            "baselines": {"status": "PENDING"}
        },
        "preserved_hashes": preserved_hashes(ROOT),
        "limitations": ["Evaluated entirely on holdout tracks without retraining. Label balance does not automatically prove semantics."]
    }
    with open(new_run / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    try:
        # 1. Prepare configuration and manifest for the new run
        config = json.loads((old_run / "config.json").read_text(encoding="utf-8"))
        config["paths"]["raw_pairs"] = str(eval_dataset)
        config["paths"]["processed"] = str(new_run / "splits")
        config["paths"]["checkpoints"] = str(new_run / "checkpoints")
        config["paths"]["results"] = str(new_run / "metrics")
        config["paths"]["workflow_cases"] = str(new_run / "workflow_cases.jsonl")
        (new_run / "metrics").mkdir(parents=True, exist_ok=True)

        with open(new_run / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        # Load data
        rows = []
        with open(eval_dataset, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))

        dataset_hash = sha256(rows)

        from cert_recovery.evidence_audit import fit_baselines
        train_ids = {json.loads(line)["pair_id"] for line in (old_run / "splits/train.jsonl").read_text().splitlines() if line.strip()}
        train_rows = [json.loads(line) for line in (old_run / "dependency_pairs.jsonl").read_text().splitlines() if line.strip()]
        fitted = fit_baselines([r for r in train_rows if r["pair_id"] in train_ids])
        global_prob, id_probs = fitted["global_probability"], fitted["identifier_probabilities"]
        pairs = json.loads(Path(args.pair_manifest).read_text())["pairs"] if args.pair_manifest else None
        if pairs is not None:
            from cert_recovery.paired_metrics import validate_pairs
            validate_pairs(rows, pairs)
        baseline_started = perf_counter()

        constant_probs = [global_prob for _ in rows]
        id_diagnostic_probs = [id_probs.get(r['derivation']['intervention']['source'], global_prob) for r in rows]

        const_rep = _prediction_report(rows, constant_probs, 0.5, config["model"]["calibration_bins"], [{}] * len(rows))
        id_rep = _prediction_report(rows, id_diagnostic_probs, 0.5, config["model"]["calibration_bins"], [{}] * len(rows))

        coverage = sum(r['derivation']['intervention']['source'] in id_probs for r in rows)
        provenance = {**fitted, "seen_identifier_rows": coverage, "total_rows": len(rows),
                      "identifier_coverage": coverage / len(rows),
                      "unseen_fallback": "training global prior; threshold 0.5",
                      "interpretation": "constant fallback is not evidence against identifier shortcuts"}
        for report, probabilities in [(const_rep, constant_probs), (id_rep, id_diagnostic_probs)]:
            report["baseline_provenance"] = provenance
            report["dataset_hash"] = dataset_hash
            report["paired_metrics"] = compute_paired_metrics(rows, probabilities, 0.5, pairs)

        with open(new_run / "metrics" / "constant_baseline.json", "w", encoding="utf-8") as f:
            json.dump(const_rep, f, indent=2)

        with open(new_run / "metrics" / "identifier_baseline.json", "w", encoding="utf-8") as f:
            json.dump(id_rep, f, indent=2)

        manifest["stages"]["baselines"] = {"status": "SUCCESS", "elapsed_seconds": perf_counter()-baseline_started}

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed
        manifest["environment"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        t0 = perf_counter()

        # 3. Evaluate Finetuned Classifier
        print("Evaluating finetuned classifier...")
        target = old_run / "checkpoints"
        selection = json.loads((target / "selection.json").read_text(encoding="utf-8"))

        if selection["checkpoint_sha256"] != sha256_file(target / "best/model.safetensors"):
            raise ValueError("Selected checkpoint bytes changed")

        model = AutoModelForSequenceClassification.from_pretrained(target / "best", local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(target / "best", local_files_only=True)

        threshold = selection["threshold"]
        test_raw = []
        test_probs = _infer(config, rows, model, tokenizer, 1, test_raw)

        paired_metrics = compute_paired_metrics(rows, test_probs, threshold, pairs)

        finetuned_report = {
            "status": "SUCCESS",
            "elapsed_seconds": perf_counter() - t0,
            "selection": selection,
            "dataset_hash": dataset_hash,
            "checkpoint_hash": selection["checkpoint_sha256"],
            "test": _prediction_report(rows, test_probs, threshold, config["model"]["calibration_bins"], test_raw),
            "paired_metrics": paired_metrics
        }

        with open(new_run / "metrics" / "finetuned_evaluation.json", "w", encoding="utf-8") as f:
            json.dump(finetuned_report, f, indent=2)

        manifest["stages"]["evaluation"] = {"status": "SUCCESS", "elapsed_seconds": perf_counter() - t0}

        # 4. Evaluate Pretrained NLI Proxy
        t1 = perf_counter()
        print("Evaluating pretrained NLI proxy...")
        model_id = config["model"]["pretrained_id"]
        revision = config["model"]["revision"]
        set_seed(42)
        tokenizer_pre = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
        model_pre = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision, trust_remote_code=False, use_safetensors=True)

        mapping = {label.lower(): int(index) for index, label in model_pre.config.id2label.items()}
        positive = mapping[config["model"]["entailment_label"]]

        pretrained_selection = json.loads((old_run / "metrics" / "pretrained_selection.json").read_text(encoding="utf-8"))
        pre_threshold = pretrained_selection["threshold"]

        pre_test_raw = []
        pre_test_probs = _infer(config, rows, model_pre, tokenizer_pre, positive, pre_test_raw)
        pre_paired_metrics = compute_paired_metrics(rows, pre_test_probs, pre_threshold, pairs)

        pretrained_report = {
            "status": "SUCCESS",
            "elapsed_seconds": perf_counter() - t1,
            "method": "pretrained NLI entailment proxy",
            "model_id": model_id, "requested_revision": revision,
            "test": _prediction_report(rows, pre_test_probs, pre_threshold, config["model"]["calibration_bins"], pre_test_raw),
            "paired_metrics": pre_paired_metrics
        }

        with open(new_run / "metrics" / "pretrained_baseline.json", "w", encoding="utf-8") as f:
            json.dump(pretrained_report, f, indent=2)

        manifest["stages"]["pretrained"] = {"status": "SUCCESS", "elapsed_seconds": perf_counter() - t1}

        # Save manifest
        manifest["execution_status"] = "SUCCESS"
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        with open(new_run / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Finalize report
        print("Finalizing HTML report...")
        report_lines = [f"<!doctype html><html lang='en'><meta charset='utf-8'><title>Frozen Evaluation</title><style>body{{font:16px system-ui;max-width:1100px;margin:30px auto;padding:20px}}table{{border-collapse:collapse}}td,th{{border:1px solid #aaa;padding:8px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}img{{max-width:100%}}</style><h1>PRELIMINARY / EXPERIMENTAL</h1>",
                  "<p>Frozen evaluation on bounded test suite. Confidence is advisory.</p><p>Artifacts: <a href='manifest.json'>manifest</a> Ã‚Â· <a href='config.json'>configuration</a> Ã‚Â· <a href='metrics/finetuned_evaluation.json'>metrics</a></p>",
                  "<h2>Execution status</h2><table><tr><th>Stage</th><th>Status</th><th>Elapsed seconds</th></tr>"]
        for stage, state in manifest["stages"].items():
            seconds = f"{state.get('elapsed_seconds'):.3f}" if 'elapsed_seconds' in state else "unmeasured"
            report_lines.append(f"<tr><td>{stage}</td><td>{state['status']}</td><td>{seconds}</td></tr>")

        report_lines.append("</table>")

        report_lines.append("<h2>Results</h2>")
        report_lines.append(f"<pre>Macro F1: {finetuned_report['test']['macro_f1']:.4f}\n{json.dumps(finetuned_report['test']['per_class'], indent=2)}</pre>")
        report_lines.append("<pre>Explicit paired metrics: " + json.dumps(paired_metrics, indent=2) + "</pre>")

        report_lines.append("<h3>Pretrained NLI Proxy</h3>")
        report_lines.append(f"<pre>Macro F1: {pretrained_report['test']['macro_f1']:.4f}\n{json.dumps(pretrained_report['test']['per_class'], indent=2)}</pre>")

        report_lines.append("<h3>Identifier Baseline (Original Training Dist)</h3>")
        report_lines.append(f"<pre>Macro F1: {id_rep['macro_f1']:.4f}\n{json.dumps(id_rep['per_class'], indent=2)}</pre>")

        with open(new_run / "report.html", "w", encoding="utf-8") as f:
            f.write("".join(report_lines))

        print("Done. Report written to:", new_run / "report.html")

    except Exception as e:
        manifest["execution_status"] = "FAILED"
        manifest["error"] = str(e)
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        with open(new_run / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        raise

if __name__ == "__main__":
    main()
