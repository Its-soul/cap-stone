"""
research_benchmark.experiment_logger
====================================
Structured telemetry and experimental logging for reproducible model benchmarking.
Maintains run metadata, configuration snapshots, dataset fingerprints,
hardware metrics, loss histories, and evaluation outcomes.
"""

import os
import json
import time
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime

class ExperimentLogger:
    """Logs and persists benchmark experiment runs to JSON and summary manifests."""
    def __init__(self, log_dir: str = "results/experiments"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    @staticmethod
    def compute_dataset_fingerprint(examples: list) -> str:
        """Computes a SHA-256 fingerprint over dataset items to guarantee split integrity."""
        h = hashlib.sha256()
        for ex in examples:
            serialized = f"{ex.get('id', '')}|{ex.get('premise', '')}|{ex.get('hypothesis', '')}|{ex.get('label', '')}"
            h.update(serialized.encode("utf-8"))
        return h.hexdigest()[:16]

    def log_run(
        self,
        model_name: str,
        group_id: str,
        task_name: str,
        training_mode: str,
        config: Dict[str, Any],
        dataset_name: str,
        dataset_fingerprint: str,
        metrics_before: Optional[Dict[str, Any]],
        metrics_after: Optional[Dict[str, Any]],
        hardware_info: Dict[str, Any],
        duration_seconds: float,
        status: str,
        error_msg: Optional[str] = None,
        checkpoint_path: Optional[str] = None
    ) -> str:
        """Saves run details to a standalone JSON file and appends to run manifest."""
        run_id = f"{group_id}_{model_name.replace('/', '_').replace(' ', '_')}_{int(time.time())}"
        record = {
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model_name": model_name,
            "group_id": group_id,
            "task_name": task_name,
            "training_mode": training_mode, # "SMOKE", "ONE_EPOCH", "MULTI_EPOCH", "INFERENCE_ONLY", "DEFERRED"
            "status": status,
            "duration_seconds": round(duration_seconds, 2),
            "config": config,
            "dataset": {
                "name": dataset_name,
                "fingerprint": dataset_fingerprint
            },
            "metrics": {
                "before": metrics_before,
                "after": metrics_after
            },
            "hardware": hardware_info,
            "checkpoint_path": checkpoint_path,
            "error_msg": error_msg
        }

        # Write individual run record
        run_file = os.path.join(self.log_dir, f"{run_id}.json")
        with open(run_file, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        # Append to summary manifest
        manifest_path = os.path.join(self.log_dir, "benchmark_manifest.jsonl")
        summary_line = {
            "run_id": run_id,
            "timestamp": record["timestamp"],
            "model_name": model_name,
            "group_id": group_id,
            "status": status,
            "training_mode": training_mode,
            "macro_f1_before": metrics_before.get("macro_f1") if metrics_before else None,
            "macro_f1_after": metrics_after.get("macro_f1") if metrics_after else None,
            "duration_seconds": record["duration_seconds"]
        }
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary_line) + "\n")

        print(f"[+] Logged experiment run: {run_id} -> {run_file}")
        return run_id

if __name__ == "__main__":
    logger = ExperimentLogger()
    print("[*] ExperimentLogger initialized at results/experiments.")
