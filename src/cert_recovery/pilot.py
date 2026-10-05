"""Bounded synthetic pilot using the existing pipelines, never run on import."""
from collections import defaultdict
import copy
import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import sys
from time import perf_counter
import traceback
import uuid

from .config import load_config
from .crypto import canonical_json
from .data_pipeline import prepare_data, write_json, load_verified_splits
from .evaluation import BASELINES, build_arithmetic_workflow, run_workflow_case, paired_cost_records
from .models import ArtifactKind
from .pinned_smoke import prepare_smoke, sha256_file

WARNING = "PRELIMINARY / EXPERIMENTAL — not final or production results."
# A family is an indivisible split group. Related paired worlds share that family.
TEMPLATES = [
    ("Snapshot {world}: input {source} has numeric value {value}.", "In {world}, claim {claim} totals exactly {bindings}; no other sources enter its sum."),
    ("The reading at {source} in world {world} is {value}.", "For {world}, certified total {claim} adds only readings {bindings}."),
    ("Record {world} stores {value} under source {source}.", "Calculation {claim} in record {world} is the sum of {bindings}, excluding other records."),
    ("Ledger {world}: {source} contains {value} units.", "Ledger result {claim} counts the units from {bindings} and no additional sources."),
    ("World {world} provides measurement {source} = {value}.", "Measurement aggregate {claim} for this world uses precisely {bindings} in its arithmetic total."),
    ("In batch {world}, the contribution identified by {source} is {value}.", "Batch claim {claim} sums contributions named {bindings}; all unlisted contributions are excluded."),
    ("Inventory {world} has count {value} for input {source}.", "Inventory summary {claim} sums counts from this complete input list: {bindings}."),
    ("Observation {world}: counter {source} reads {value}.", "Counter total {claim} is defined by adding {bindings}, without any other counters."),
    ("Entry {source} in journal {world} records the integer {value}.", "Journal output {claim} equals the sum over exactly these entries: {bindings}."),
    ("Register {world} assigns amount {value} to item {source}.", "Register claim {claim} adds amounts assigned to {bindings}; omitted items do not contribute."),
    ("At location {world}, sensor {source} reports {value}.", "Sensor result {claim} totals readings from {bindings} only."),
    ("Account {world} contains balance {value} at source {source}.", "Account total {claim} uses this exhaustive list of balances in its sum: {bindings}."),
]


def synthetic_pairs(costs, seed=42):
    """Labels are declared source-sum ancestry, witnessed by a +1 intervention.

    Not annotations of real causal necessity. No six-case NLI text is used.
    """
    rng = random.Random(seed)
    rows = []
    for family, (premise_template, hypothesis_template) in enumerate(TEMPLATES):
        for world_index in range(4):
            world = f"family{family:02d}-world{world_index}"
            topology = "cascade" if world_index % 2 else "independent"
            values = [rng.randrange(-20, 40) for _ in range(3)]
            case = {"world_id": world, "topology": topology, "sources": [{"value": v} for v in values]}
            system, _ = build_arithmetic_workflow(case, costs)
            target_index = rng.randrange(2)
            claim = system.latest(f"Q{target_index}")
            roots = sorted(node.ref.artifact_id for node in system.provenance(claim.ref)
                           if node.kind == ArtifactKind.DEPENDENCY)
            candidates = [(rng.choice(roots), 1),
                          (rng.choice([f"D{i}" for i in range(3) if f"D{i}" not in roots]), 0)]
            # Same lexical context for both labels; IDs and numbers are not label encodings.
            for source_id, label in candidates:
                index = int(source_id[1:])
                witness, _ = build_arithmetic_workflow(case, costs)
                before = witness.latest(claim.ref.artifact_id).payload["value"]
                change = witness.change_dependency(source_id, {"value": values[index] + 1})
                witness.execute_recovery(witness.plan_recovery(change))
                after = witness.latest(claim.ref.artifact_id).payload["value"]
                if (after != before) != bool(label):
                    raise AssertionError("Synthetic label failed its deterministic intervention witness")
                context = dict(world=world, source=source_id, value=values[index],
                               claim=claim.ref.artifact_id, bindings=", ".join(roots))
                rows.append({"pair_id": f"{world}-{source_id}-{claim.ref.artifact_id}",
                             "world_id": world, "template_family": f"family{family:02d}",
                             "premise": premise_template.format(**context),
                             "hypothesis": hypothesis_template.format(**context), "label": label,
                             "source": "synthetic source-sum pilot v1; not domain annotation",
                             "derivation": {"rule": "source in declared transitive source-sum ancestry",
                                            "bindings": roots, "topology": topology,
                                            "intervention": {"source": source_id, "delta": 1,
                                                             "before": before, "after": after}}})
    return rows


def workflow_cases():
    return [{"world_id": f"{topology}-{kind}-{index}", "topology": topology,
             "sources": [{"value": 2}, {"value": -3}, {"value": 7}],
             "change": {"source_index": index, "kind": kind, "value": 9}}
            for topology in ("independent", "cascade")
            for kind in ("update", "remove", "untrusted") for index in (0, 2)]


def audit_synthetic_identifiers(rows):
    """Expose source-name shortcuts; never filter, relabel or tune from this audit."""
    counts = defaultdict(lambda:[0,0])
    for row in rows:
        counts[row["derivation"]["intervention"]["source"]][row["label"]] += 1
    counts = dict(sorted(counts.items()))
    single_class = [name for name,values in counts.items() if not all(values)]
    return {"warning":WARNING,"scope":"synthetic source identifier contingency, not a model evaluation",
            "count_order":["independent","depends_on"],"source_label_counts":counts,
            "single_class_source_ids":single_class,
            "finding":"D2 never appears in selected Q0/Q1 support. Source-name cues can confound scores despite disjoint template families.",
            "action":"Preserve this run. Use a new frozen set with randomized aliases/source roles before further tuning."}


def preserved_hashes(root):
    files = [root / "baseline_tests.json", *sorted((root / "_private/research").rglob("*.json")),
             *sorted((root / "_private/research/notebooks").glob("*.ipynb"))]
    return {str(path.relative_to(root)): sha256_file(path) for path in files if path.is_file()}


def create_run(root):
    root = Path(root).resolve()
    run = root / "_private/runs" / (datetime.now(timezone.utc).strftime("pilot_%Y%m%dT%H%M%S_%fZ_") + uuid.uuid4().hex[:8])
    run.mkdir(parents=True)
    config = load_config(root / "config/config.yaml")
    spec = prepare_smoke(root, "deberta")["model"]
    config["execution"] = {key: True for key in config["execution"]}
    config["model"].update(pretrained_id=spec["model_id"], revision=spec["revision"], max_length=128,
                           inference_batch_size=2)
    config["training"].update(batch_size=2, epochs=2, save_total_limit=1)
    config["data"]["group_key"] = "template_family"
    config["paths"].update(raw_pairs=str(run / "dependency_pairs.jsonl"), processed=str(run / "splits"),
                           checkpoints=str(run / "checkpoints"), results=str(run / "metrics"),
                           workflow_cases=str(run / "workflow_cases.jsonl"))
    write_json(run / "config.json", config)
    revision = subprocess.run(["git","rev-parse","HEAD"],cwd=root,text=True,capture_output=True)
    status = subprocess.run(["git","status","--short"],cwd=root,text=True,capture_output=True)
    manifest = {"schema_version": 1, "run_id": run.name, "warning": WARNING,
                "started_utc": datetime.now(timezone.utc).isoformat(), "study": "controlled synthetic pilot",
                "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
                "git_status": status.stdout if status.returncode == 0 else None,
                "git_provenance": "checkout" if revision.returncode == 0 else "unknown; exported source bytes are hashed",
                "code_sha256": {str(p.relative_to(root)): sha256_file(p) for p in sorted((root / "src").rglob("*.py"))},
                "runtime": {"python": sys.version, "platform": platform.platform(), "processor": platform.processor()},
                "preserved_hashes": preserved_hashes(root),
                "budget": {"training_runs": 1, "epochs": 2, "model_stage_timeout_seconds": 1800,
                           "smoke_pairs_per_model": 6, "system_cases": 12, "system_repetitions": 3},
                "stages": {stage: {"status": "NOT RUN"} for stage in
                           ("data", "system", "deberta", "minilm_fallback", "pretrained", "training", "evaluation")},
                "limitations": ["Synthetic labels reflect declared arithmetic support, not real causal necessity.",
                                 "One training seed; no hyperparameter search or production/generalization claim.",
                                 "Held-out template families, 16 test rows; calibration summaries are descriptive.",
                                 "Direct model identity cannot attest historical Space weights."]}
    write_json(run / "manifest.json", manifest)
    return run


def execute_stage(run, stage):
    run = Path(run)
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    start = perf_counter()
    state = {"status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(), "warning": WARNING}
    state["code_sha256"] = {str(p.relative_to(Path(config["project_root"]))):sha256_file(p)
                            for p in sorted((Path(config["project_root"])/"src").rglob("*.py"))}
    for relative in state["code_sha256"]:
        destination = run / "source" / stage / relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(Path(config["project_root"])/relative,destination)
    state_path = run / (stage + ".status.json")
    if state_path.exists():
        raise FileExistsError("Stage already attempted; preserve its result and choose a new run")
    write_json(state_path, state)
    try:
        if stage == "data":
            rows = synthetic_pairs(config["recovery"])
            write_json(run / "data_audit.json",audit_synthetic_identifiers(rows))
            Path(config["paths"]["raw_pairs"]).write_text("".join(canonical_json(row)+"\n" for row in rows), encoding="utf-8")
            cases = workflow_cases()
            Path(config["paths"]["workflow_cases"]).write_text("".join(canonical_json(case)+"\n" for case in cases), encoding="utf-8")
            state["split_manifest"] = prepare_data(config, manual=True)
            splits, _ = load_verified_splits(config)
            if any({row["label"] for row in items} != {0,1} for items in splits.values()):
                raise ValueError("Every pilot split needs both labels")
            state["raw_data_sha256"] = sha256_file(config["paths"]["raw_pairs"])
        elif stage == "system":
            records = []
            for repetition in range(3):
                for case in workflow_cases():
                    for baseline in BASELINES:
                        started = perf_counter()
                        try:
                            row = run_workflow_case(copy.deepcopy(case), baseline, config, manual=True)
                            row.update(repetition=repetition, outer_elapsed_seconds=perf_counter()-started)
                        except Exception as error:
                            row = {"world_id": case["world_id"], "baseline": baseline, "repetition": repetition,
                                   "status": "ERROR", "error": str(error), "outer_elapsed_seconds": perf_counter()-started}
                        records.append(row)
            write_json(run / "metrics/workflow_benchmark.json", {"warning": WARNING, "records": records,
                       "paired_cost": paired_cost_records([r for r in records if r["status"] != "ERROR"])})
            state["attempts"] = len(records)
            state["failed_attempts"] = sum(row["status"] == "ERROR" for row in records)
            if state["failed_attempts"]:
                raise RuntimeError("System attempts failed; see captured records")
        else:
            import torch
            from importlib.metadata import version
            torch.set_num_threads(4)
            state["runtime"] = {"device": "cuda" if torch.cuda.is_available() else "cpu", "threads": 4,
                                "packages": {p: version(p) for p in ("torch", "transformers", "datasets", "accelerate", "huggingface-hub", "numpy")}}
            if stage in ("deberta", "minilm_fallback"):
                from .pinned_smoke import run_pinned_smoke
                result = run_pinned_smoke(config, stage, manual=True, all_cases=True)
                state["result_run_id"] = result["run_id"]
                state["matches"] = sum(p["matches_expected_label"] for p in result["predictions"])
            else:
                from .model_pipeline import run_finetuning, run_model_evaluation, run_pretrained_baseline
                if stage == "pretrained":
                    run_pretrained_baseline(config, manual=True, include_test=True, selection_frozen=True)
                elif stage == "training":
                    run_finetuning(config, manual=True)
                elif stage == "evaluation":
                    run_model_evaluation(config, manual=True, selection_frozen=True)
                else:
                    raise ValueError("Unknown stage")
        state["status"] = "SUCCESS"
    except Exception as error:
        state.update(status="ERROR", error={"type": type(error).__name__, "message": str(error),
                                             "traceback": traceback.format_exc()})
    finally:
        state.update(elapsed_seconds=perf_counter()-start, finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(state_path, state)
    return state


def csv_rows(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows)


def finalize(run):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    for stage in manifest["stages"]:
        path = run / (stage + ".status.json")
        manifest["stages"][stage] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status":"NOT RUN"}
    # Explicit runtime-fix attempts retain the first failure; never overwrite its status/log.
    manifest["stage_attempts"] = {}
    for attempt in sorted(run.glob("attempt_*/attempt.json")):
        provenance = json.loads(attempt.read_text(encoding="utf-8"))
        for stage in ("training","evaluation"):
            path = attempt.parent / (stage+".status.json")
            if path.exists():
                old = manifest["stages"][stage]
                latest = json.loads(path.read_text(encoding="utf-8"))
                manifest["stage_attempts"].setdefault(stage,[] if old["status"] == "NOT RUN" else [old]).append({**latest,"attempt_provenance":provenance})
                manifest["stages"][stage] = latest
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    root = Path(config["project_root"])
    if manifest["preserved_hashes"] != preserved_hashes(root):
        raise AssertionError("Historical results/notebooks or six-case dataset changed")
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(run / "manifest.json", manifest)
    report = [f"<!doctype html><html lang='en'><meta charset='utf-8'><title>Synthetic pilot</title><style>body{{font:16px system-ui;max-width:1100px;margin:30px auto;padding:20px}}table{{border-collapse:collapse}}td,th{{border:1px solid #aaa;padding:8px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}img{{max-width:100%}}</style><h1>{html.escape(WARNING)}</h1>",
              "<p>Controlled source-sum synthetic pilot. Confidence is advisory; deterministic checks retain authority.</p><p>Artifacts: <a href='manifest.json'>manifest</a> · <a href='config.json'>configuration</a> · <a href='summary.json'>metrics</a> · <a href='findings.md'>findings</a></p>",
              "<h2>Execution status</h2><table><tr><th>Stage</th><th>Status</th><th>Elapsed seconds</th></tr>"]
    for stage, state in manifest["stages"].items():
        measured = state.get("elapsed_seconds")
        seconds = f"{measured:.3f}" if measured is not None else "unmeasured"
        report.append(f"<tr><td>{stage}</td><td>{state['status']}</td><td>{seconds}</td></tr>")
        if state.get("error"):
            report.append(f"<tr><td colspan='3'><pre>{html.escape(state['error']['message'])}</pre></td></tr>")
        if stage in manifest["stage_attempts"]:
            attempts = [{"status":a["status"],"elapsed_seconds":a.get("elapsed_seconds"),
                         "error":a.get("error",{}).get("message"),
                         "reason":a.get("attempt_provenance",{}).get("reason")} for a in manifest["stage_attempts"][stage]]
            report.append("<tr><td colspan='3'><details><summary>Preserved attempts</summary><pre>"+html.escape(json.dumps(attempts,indent=2))+"</pre></details></td></tr>")
    report.append("</table>")
    if (run/"data_audit.json").exists():
        audit = json.loads((run/"data_audit.json").read_text(encoding="utf-8"))
        report.append("<h2>Dataset weakness</h2><p>"+html.escape(audit["finding"])+"</p><p>"+html.escape(audit["action"])+"</p>")
    summary = {"warning": WARNING, "models": {}, "system": {}}
    for name in ("pretrained_baseline", "finetuned_evaluation"):
        path = run / "metrics" / (name + ".json")
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        test = record["test"]
        summary["models"][name] = {k:v for k,v in test.items() if k not in ("predictions", "examples")}
        # Derived reporting correction only: preserve captured raw result JSON.
        for label,count in test["class_counts"].items():
            summary["models"][name]["per_class"][label]["support"] = count
        csv_rows(run / (name + ".predictions.csv"), test["predictions"])
        write_json(run / (name + ".errors.json"), {"warning": WARNING, "examples": test["examples"]})
        report.append(f"<h2>{name} — same frozen synthetic test set</h2><table><tr><th>Rows</th><th>Accuracy</th><th>Macro F1</th><th>Missed dependencies</th><th>Brier / ECE (descriptive)</th></tr><tr><td>{test['classification']['support']}</td><td>{test['classification']['accuracy']:.4f}</td><td>{test['macro_f1']:.4f}</td><td>{test['false_negative_dependencies']}</td><td>{test['calibration']['brier']:.4f} / {test['calibration']['ece']:.4f}</td></tr></table><details><summary>Class counts, confusion, per-class metrics and threshold</summary><pre>{html.escape(json.dumps(summary['models'][name],indent=2))}</pre></details>")
    path = run / "metrics/workflow_benchmark.json"
    if path.exists():
        rows = json.loads(path.read_text(encoding="utf-8"))["records"]
        csv_rows(run / "workflow_records.csv", rows)
        csv_rows(run / "paired_work_units.csv", paired_cost_records([r for r in rows if r["status"] != "ERROR"]))
        for baseline in BASELINES:
            items = [r for r in rows if r["baseline"] == baseline and r["status"] != "ERROR"]
            if items:
                summary["system"][baseline] = {"attempts": len(items), "safe_commits": sum(r["safe_commits"] for r in items),
                    "commits": sum(r["total_commits"] for r in items), "tasks": sum(r["total_tasks"] for r in items),
                    "blocked_tasks": sum(r["blocked_tasks"] for r in items),
                    "invalidated_certificates": sum(len(r["impact"]["direct_certificates"])+len(r["impact"]["transitive_certificates"]) if r["impact"] else 0 for r in items),
                    "recovery_restored_artifacts": sum(len(r["recovery"]["restored"]) if r["recovery"] else 0 for r in items),
                    "mean_policy_seconds": statistics.mean(r["latency_seconds"] for r in items),
                    "mean_total_seconds": statistics.mean(r["outer_elapsed_seconds"] for r in items),
                    "configured_work_units": sum(r["configured_work_units"] for r in items)}
        report.append("<h2>Paired system results: 12 cases × 3 repetitions × 4 policies</h2><p>Work units are configured estimates, not money or measured latency. Cheap blocking is not successful recovery. Timings include failed/blocked policy work; total includes setup.</p><pre>" + html.escape(json.dumps(summary["system"],indent=2)) + "</pre>")
    for path in sorted((run/"metrics").glob("pinned_smoke_*.json")):
        if path.name.endswith(".prepared.json"):
            continue
        captured = json.loads(path.read_text(encoding="utf-8"))
        report.append("<h2>Direct pinned "+html.escape(captured["profile"])+" — unchanged NLI smoke pairs</h2><p>Separate from historical Space output. Six cases cannot establish accuracy or calibration.</p><details><summary>Verified weights, tokenizer and runtime</summary><pre>"+html.escape(json.dumps({k:captured.get(k) for k in ("loaded_weights","tokenizer_files","runtime")},indent=2))+"</pre></details><table><tr><th>Case</th><th>Exact directional input</th><th>Expected</th><th>Observed</th><th>Probability</th></tr>")
        for prediction in captured["predictions"]:
            expected = prediction["case"]["expected_label"]
            observed = prediction["canonical_prediction"]
            pair = prediction["case"]["input"]
            text = html.escape(pair["premise"])+" → "+html.escape(pair["hypothesis"])
            report.append(f"<tr><td>{prediction['case']['id']}</td><td>{text}</td><td>{expected}</td><td>{observed}</td><td>{prediction['class_probabilities'][observed]:.6f}</td></tr>")
        report.append("</table>")
    write_json(run / "summary.json", summary)
    csv_rows(run / "model_metrics.csv", [{"method": k, "accuracy": v["classification"]["accuracy"],
              "macro_f1": v["macro_f1"], "false_negative_dependencies": v["false_negative_dependencies"]} for k,v in summary["models"].items()])
    # Plots are created only from measured artifacts; no dummy curves for unrun stages.
    try:
        from .pilot_plots import make_plots
        for path in make_plots(run, summary):
            import base64
            report.append(f"<h2>{html.escape(path.stem)}</h2><img alt='{html.escape(path.stem)}' src='data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}'>")
    except ImportError:
        report.append("<p>Plots NOT GENERATED: matplotlib unavailable; measured tables remain available.</p>")
    report.append("<h2>Limitations</h2><ul>" + "".join("<li>"+html.escape(x)+"</li>" for x in manifest["limitations"]) + "</ul></html>")
    if (run/"ANALYSIS.md").exists():
        report.insert(2,"<p><a href='ANALYSIS.md'>Analysis, failures and next development</a></p>")
    (run / "report.html").write_text("\n".join(report), encoding="utf-8")
    (run / "findings.md").write_text(WARNING + "\n\n" + json.dumps(summary,indent=2) + "\n\n" + "\n".join(manifest["limitations"]) + "\n", encoding="utf-8")
    return summary
