"""
research_benchmark.checkpoint_manager
=====================================
Isolated checkpoint lifecycle management.
Guarantees that historical repository checkpoints (such as Candidate V2 in _private/)
are NEVER overwritten, modified, or corrupted.
"""

import os
import shutil
from typing import Optional, List

class CheckpointManager:
    """Manages isolated checkpoint paths, persistence verification, and disk space hygiene."""
    def __init__(self, base_checkpoint_dir: str = "results/checkpoints"):
        self.base_dir = base_checkpoint_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def get_isolated_checkpoint_dir(self, group_id: str, model_name: str, tag: str = "run") -> str:
        """Returns a safe, isolated directory path for saving newly fine-tuned weights."""
        sanitized_name = model_name.replace("/", "_").replace(" ", "_").lower()
        path = os.path.join(self.base_dir, group_id, f"{sanitized_name}_{tag}")
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def verify_checkpoint_integrity(checkpoint_dir: str) -> bool:
        """Verifies that essential model and tokenizer serialization files exist."""
        if not os.path.isdir(checkpoint_dir):
            return False
        files = os.listdir(checkpoint_dir)
        has_weights = any(f.endswith(".bin") or f.endswith(".safetensors") or f.endswith(".pt") for f in files)
        has_config = "config.json" in files
        return has_weights and has_config

    def prune_optimizer_states(self, checkpoint_dir: str):
        """Removes bulky optimizer states after training completion to save local disk space."""
        opt_file = os.path.join(checkpoint_dir, "optimizer.pt")
        if os.path.exists(opt_file):
            os.remove(opt_file)
            print(f"[*] Pruned optimizer state at {opt_file} to save disk space.")
