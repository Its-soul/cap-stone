"""
research_benchmark.trainers
===========================
Standardized training loops for:
1. Laptop CPU 1-Epoch Feasibility (Resource-bounded, memory-guarded, float32)
2. NVIDIA Laboratory Multi-Epoch Training (Mixed-precision, early stopping, multi-seed)
"""

import os
import sys
import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
try:
    from torch.optim import AdamW
except ImportError:
    from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from .hardware import configure_cpu_safety_limits, check_memory_headroom, get_current_process_memory_mb
from .evaluation import compute_classification_metrics, compute_dependency_metrics

# ---------------------------------------------------------------------------
# Simple in-memory paired text dataset
# ---------------------------------------------------------------------------
class SimpleTextPairDataset(Dataset):
    def __init__(self, tokenizer, pairs: List[Tuple[str, str]], labels: List[int], max_length: int = 128):
        self.encodings = tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

# ---------------------------------------------------------------------------
# 1. Laptop CPU 1-Epoch Trainer
# ---------------------------------------------------------------------------
class LaptopOneEpochTrainer:
    """
    Resource-guarded, CPU-safe 1-epoch feasibility trainer.
    Purpose: Verify dataset ingestion, loss computation, gradient backward pass,
    optimizer stepping, and checkpoint serialization without destabilizing laptop OS.
    """
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        lr: float = 2e-5,
        batch_size: int = 2,
        max_seq_len: int = 128,
        weight_decay: float = 0.01,
        max_memory_mb: float = 3500.0,
        output_dir: str = "results/checkpoints/laptop_one_epoch"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.lr = lr
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.weight_decay = weight_decay
        self.max_memory_mb = max_memory_mb
        self.output_dir = output_dir
        self.device = torch.device("cpu")
        
        # Enforce CPU safety
        configure_cpu_safety_limits(num_threads=4)

    def train_one_epoch(
        self,
        train_examples: List[Dict[str, Any]],
        val_examples: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Executes exactly 1 epoch over train_examples on CPU.
        Returns detailed telemetry, loss curve, duration, and post-epoch validation metrics.
        """
        print(f"\n[+] Starting Laptop 1-Epoch Feasibility Run on {len(train_examples)} examples (Batch: {self.batch_size}, LR: {self.lr})...")
        start_time = time.time()
        
        if not check_memory_headroom(min_free_mb=500.0):
            return {
                "status": "OOM_ABORT",
                "error": "Insufficient host RAM headroom before launch.",
                "duration_seconds": 0.0
            }

        # Prepare dataset & dataloader
        pairs = [(ex["premise"], ex["hypothesis"]) for ex in train_examples]
        labels = [int(ex["label"]) for ex in train_examples]
        dataset = SimpleTextPairDataset(self.tokenizer, pairs, labels, max_length=self.max_seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.to(self.device)
        self.model.train()
        
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        total_steps = len(loader)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, int(total_steps * 0.1)), num_training_steps=total_steps)

        losses = []
        step_times = []
        initial_ram_mb = get_current_process_memory_mb()
        peak_ram_mb = initial_ram_mb

        try:
            for step, batch in enumerate(loader):
                t_step_start = time.time()
                
                # Dynamic RAM safeguard
                current_ram = get_current_process_memory_mb()
                if current_ram > peak_ram_mb:
                    peak_ram_mb = current_ram
                if current_ram > self.max_memory_mb:
                    raise MemoryError(f"Process memory exceeded safety ceiling: {current_ram:.1f} MB > {self.max_memory_mb} MB")

                optimizer.zero_grad()
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**inputs)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                loss_val = float(loss.item())
                losses.append(loss_val)
                step_times.append(time.time() - t_step_start)

                if (step + 1) % max(1, (total_steps // 5)) == 0 or (step + 1) == total_steps:
                    print(f"    Step {step + 1}/{total_steps} - Loss: {loss_val:.4f} - RAM: {current_ram:.1f} MB")

            total_duration = time.time() - start_time
            print(f"[+] 1-Epoch completed successfully in {total_duration:.2f}s! Mean Loss: {np.mean(losses):.4f}")

            # Optional post-epoch validation
            val_metrics = None
            if val_examples:
                val_metrics = self.evaluate(val_examples)
                print(f"[+] Post-Epoch Validation Macro-F1: {val_metrics['macro_f1']:.4f} (Accuracy: {val_metrics['accuracy']:.4f})")

            # Save checkpoint smoke test
            os.makedirs(self.output_dir, exist_ok=True)
            self.model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            print(f"[+] Saved checkpoint to: {self.output_dir}")

            return {
                "status": "SUCCESS",
                "duration_seconds": total_duration,
                "total_steps": total_steps,
                "mean_loss": float(np.mean(losses)),
                "loss_history": losses,
                "peak_ram_mb": peak_ram_mb,
                "validation_metrics": val_metrics,
                "checkpoint_path": self.output_dir
            }

        except Exception as e:
            print(f"[!] Laptop 1-Epoch Failed: {e}")
            return {
                "status": "FAILED",
                "error": str(e),
                "duration_seconds": time.time() - start_time,
                "peak_ram_mb": peak_ram_mb
            }

    def evaluate(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluates model over examples and returns classification metrics."""
        self.model.eval()
        pairs = [(ex["premise"], ex["hypothesis"]) for ex in examples]
        true_labels = [int(ex["label"]) for ex in examples]
        dataset = SimpleTextPairDataset(self.tokenizer, pairs, true_labels, max_length=self.max_seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        preds = []
        with torch.no_grad():
            for batch in loader:
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**inputs)
                batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
                preds.extend(batch_preds)

        return compute_classification_metrics(true_labels, preds)

# ---------------------------------------------------------------------------
# 2. NVIDIA Laboratory Multi-Epoch Trainer
# ---------------------------------------------------------------------------
class NVIDIALabMultiEpochTrainer:
    """
    Extensive multi-epoch trainer engineered for NVIDIA GPU servers.
    Supports mixed precision (bf16/fp16), early stopping, and multi-seed evaluation.
    """
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        epochs: int = 5,
        lr: float = 2e-5,
        batch_size: int = 16,
        max_seq_len: int = 512,
        mixed_precision: str = "bf16",
        patience: int = 2,
        output_dir: str = "results/checkpoints/nvidia_lab"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.mixed_precision = mixed_precision
        self.patience = patience
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_multi_epoch(
        self,
        train_examples: List[Dict[str, Any]],
        val_examples: List[Dict[str, Any]],
        seed: int = 42
    ) -> Dict[str, Any]:
        """Runs multi-epoch training with early stopping on validation macro-F1."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        print(f"\n[+] NVIDIA Lab Training: {self.epochs} epochs on {self.device} (Seed: {seed}, Precision: {self.mixed_precision})...")
        start_time = time.time()

        pairs = [(ex["premise"], ex["hypothesis"]) for ex in train_examples]
        labels = [int(ex["label"]) for ex in train_examples]
        dataset = SimpleTextPairDataset(self.tokenizer, pairs, labels, max_length=self.max_seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.to(self.device)
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        total_steps = len(loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

        best_val_f1 = -1.0
        patience_counter = 0
        history = []

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_losses = []
            for batch in loader:
                optimizer.zero_grad()
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                
                # Autocast if CUDA
                if self.device.type == "cuda" and self.mixed_precision == "bf16":
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = self.model(**inputs)
                        loss = outputs.loss
                else:
                    outputs = self.model(**inputs)
                    loss = outputs.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                epoch_losses.append(float(loss.item()))

            # Validation
            val_metrics = self.evaluate(val_examples)
            val_f1 = val_metrics["macro_f1"]
            mean_loss = float(np.mean(epoch_losses))
            print(f"    Epoch {epoch}/{self.epochs} - Loss: {mean_loss:.4f} - Val F1: {val_f1:.4f} (Acc: {val_metrics['accuracy']:.4f})")

            history.append({
                "epoch": epoch,
                "train_loss": mean_loss,
                "val_macro_f1": val_f1,
                "val_accuracy": val_metrics["accuracy"]
            })

            # Checkpoint & Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                best_ckpt_dir = os.path.join(self.output_dir, f"best_seed_{seed}")
                os.makedirs(best_ckpt_dir, exist_ok=True)
                self.model.save_pretrained(best_ckpt_dir)
                self.tokenizer.save_pretrained(best_ckpt_dir)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"[*] Early stopping triggered at epoch {epoch} (Patience: {self.patience})")
                    break

        return {
            "status": "SUCCESS",
            "seed": seed,
            "best_val_macro_f1": best_val_f1,
            "total_duration_seconds": time.time() - start_time,
            "epoch_history": history
        }

    def evaluate(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.model.eval()
        pairs = [(ex["premise"], ex["hypothesis"]) for ex in examples]
        true_labels = [int(ex["label"]) for ex in examples]
        dataset = SimpleTextPairDataset(self.tokenizer, pairs, true_labels, max_length=self.max_seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        preds = []
        with torch.no_grad():
            for batch in loader:
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**inputs)
                preds.extend(torch.argmax(outputs.logits, dim=-1).cpu().tolist())

        return compute_classification_metrics(true_labels, preds)
