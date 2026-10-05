"""Run a fixed NLI smoke dataset through a remote Space and preserve each run."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
import uuid

from dotenv import load_dotenv

from remote_model_client import (
    PROFILES, RESULT_SCHEMA_VERSION, RemoteModelClient, RemoteModelError,
    create_case_record, redact, utc_now,
)


ROOT = Path(__file__).resolve().parent


def load_dataset(path):
    raw = path.read_bytes()
    dataset = json.loads(raw)
    if dataset.get("schema_version") != 1 or dataset.get("task") != "natural_language_inference":
        raise ValueError("Expected schema version 1 and natural_language_inference task")
    cases = dataset.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Dataset must contain at least one case")
    ids = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"] or case["id"] in ids:
            raise ValueError("Every case needs a unique nonempty string ID")
        ids.add(case["id"])
        pair = case.get("input")
        if not isinstance(pair, dict) or set(pair) != {"premise", "hypothesis"}:
            raise ValueError("Case input must contain exactly premise and hypothesis")
        if any(not isinstance(text, str) or not text.strip() for text in pair.values()):
            raise ValueError("Premise and hypothesis must be nonempty text")
        if case.get("expected_label") not in {"entailment", "contradiction", "neutral"}:
            raise ValueError("Expected labels must be NLI labels")
    return dataset, hashlib.sha256(raw).hexdigest()


def create_run_file(directory):
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = directory / f"baseline_{stamp}_{uuid.uuid4().hex[:8]}.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump({"status": "RUNNING", "started_at": utc_now()}, stream)
    return path


def save_current_run(path, document, token=None):
    # Update only the file exclusively allocated to this active run. Never load
    # or modify earlier runs. Atomic replacement preserves the last snapshot.
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(redact(document, token), stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args):
    token = os.getenv("HF_TOKEN", "").strip() or None
    path = create_run_file(args.results_dir)
    document = {"schema_version": RESULT_SCHEMA_VERSION, "run_id": path.stem, "started_at": utc_now(),
                "status": "RUNNING", "dataset": {"filename": args.dataset.name},
                "selection": {"profile": args.profile, "alternative_model_opt_in": args.allow_alternative_model},
                "client_versions": {name: version(name) for name in ("gradio_client", "huggingface-hub", "python-dotenv")},
                "python_version": sys.version.split()[0], "results": [], "error": None,
                "note": "Remote NLI smoke testing only; does not test the project's DeBERTa checkpoint or binary dependency classifier."}
    remote = RemoteModelClient(args.profile, token=token, timeout=args.timeout)
    selected = []
    start = perf_counter()
    try:
        dataset, digest = load_dataset(args.dataset)
        selected = dataset["cases"][:args.limit] if args.limit else dataset["cases"]
        document["dataset"].update(dataset_id=dataset["dataset_id"], sha256=digest,
                                   total_cases=len(dataset["cases"]), selected_ids=[case["id"] for case in selected],
                                   is_full_dataset=len(selected) == len(dataset["cases"]),
                                   purpose=dataset["description"])
        document["remote"] = remote.metadata
        save_current_run(path, document, token)
        document["remote"] = remote.connect(allow_alternative_model=args.allow_alternative_model)
        document["quota_before"] = remote.quota()
        save_current_run(path, document, token)
        quota = document["quota_before"]
        if quota["status"] == "AVAILABLE" and quota["remaining_gpu_seconds"] <= 0:
            raise RemoteModelError("Included ZeroGPU quota is exhausted; no inference submitted")
        for case in selected:
            result = remote.infer(case, run_id=document["run_id"])
            document["results"].append(result)
            save_current_run(path, document, token)
            print(f"{case['id']}: {result['execution_status']}")
            # No retries and no silent model switching: avoid burning quota or
            # mixing incomparable models in one baseline run.
            if result["execution_status"] != "SUCCESS":
                document["error"] = result["error"]
                break
        document["quota_after"] = remote.quota()
        document["status"] = "SUCCESS" if len(document["results"]) == len(selected) and all(
            row["execution_status"] == "SUCCESS" for row in document["results"]
        ) else "FAILED"
    except KeyboardInterrupt:
        document["status"] = "INTERRUPTED"
        document["error"] = {"type": "KeyboardInterrupt", "stage": "run", "code": "interrupted",
                             "message": "Interrupted; a submitted remote job may still be running"}
    except Exception as error:
        document["status"] = "FAILED"
        document["error"] = {"type": type(error).__name__, "stage": "run", "code": "run_error",
                             "message": redact(str(error), token)}
    finally:
        document["remote"] = remote.metadata
        completed_ids = {row["test_id"] for row in document["results"]}
        for case in selected:
            if case["id"] not in completed_ids:
                result = create_case_record(args.profile, case, document["run_id"])
                result["error"] = document["error"]
                document["results"].append(result)
        document["completed_at"] = utc_now()
        document["elapsed_seconds"] = perf_counter() - start
        document["summary"] = dict(Counter(row["execution_status"] for row in document["results"]))
        document["classification_summary"] = dict(Counter(row["classification_status"] for row in document["results"]))
        save_current_run(path, document, token)
        try:
            remote.close()
        except Exception as error:
            document["close_error"] = redact(str(error), token)
            save_current_run(path, document, token)
    print(f"Run status: {document['status']}")
    if document["error"]:
        print(f"Error: {document['error']['message']}")
    print(f"Saved: {path}")
    return 0 if document["status"] == "SUCCESS" else 1


def main():
    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "baseline_tests.json")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--profile", choices=PROFILES, default=os.getenv("REMOTE_PROFILE", "nli-zero-gpu"))
    parser.add_argument("--allow-alternative-model", action="store_true",
                        help="Acknowledge that reviewed Spaces do not serve the project's configured model")
    parser.add_argument("--limit", type=int, help="Run a deterministic prefix for a smoke request; omit for the full fixed dataset")
    parser.add_argument("--timeout", type=float, default=os.getenv("REMOTE_TIMEOUT_SECONDS", "120"))
    args = parser.parse_args()
    if args.profile not in PROFILES:
        parser.error("REMOTE_PROFILE must name a reviewed profile")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
