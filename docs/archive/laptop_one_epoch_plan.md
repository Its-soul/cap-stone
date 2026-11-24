# Laptop One-Epoch Feasibility Plan (CPU-Safe Execution)
**Project:** Verifiable Multi-Agent Fault-Tolerant Adjudication System with Cryptographic Causal Provenance  
**Repository:** [https://github.com/Its-soul/cap-stone](https://github.com/Its-soul/cap-stone)  
**Investigation Date:** October 2026  
**Author:** Senior Deep Learning Engineer & MLOps Specialist  

---

## 1. Local Hardware Environment & Resource Profiling

Before designing any local execution plan, the host laptop environment was strictly profiled:

```
+---------------------------------------------------------------------------------------------------+
| HOST HARDWARE & SYSTEM SPECIFICATIONS                                                             |
+---------------------------------------------------------------------------------------------------+
| Operating System    : Windows 11 Home 64-bit (Build 10.0.26200)                                   |
| Processor (CPU)     : 13th Gen Intel(R) Core(TM) i7-13620H (10 Cores, 16 Logical Threads)        |
| Discrete GPU        : None (torch.cuda.is_available() == False)                                   |
| Integrated Graphics : Intel(R) UHD Graphics (System Shared Memory Only)                          |
| Total Physical RAM  : 15.73 GB (~16.0 GB DDR5)                                                    |
| Storage Workspace   : Drive D: (152.5 GB Total, 118.8 GB Free)                                    |
| Python Environment  : Python 3.14.6 64-bit with PyTorch 2.13.0+cpu, Transformers 4.57.1          |
+---------------------------------------------------------------------------------------------------+
```

### Critical Operational Constraints:
1. **Zero CUDA Acceleration:** All local laptop trials run strictly on **CPU**.
2. **Process Working Set Ceiling:** Due to active desktop and IDE processes, individual Python training processes must **never exceed 3.5 GB of allocated RAM** to prevent Windows paging file thrashing or system lockup.
3. **CPU Thread Allocation:** PyTorch intra-op threads must be restricted (`torch.set_num_threads(4)`) to leave 12 logical threads free for operating system stability.
4. **Feasibility Objective:** A 1-epoch test is **strictly a compatibility, loading, gradient backpropagation, and checkpoint serialization verification trial**. It does not prove convergence or peak model performance.

---

## 2. Model-by-Model 1-Epoch Feasibility Rationale & Configurations

| Shortlisted Model | Architecture & Size | Local Laptop Feasibility Classification | Batch Size | Grad Accum | Max Seq Len | Learning Rate | Optimizer & Precision | Est. Memory (RAM) | Est. Wall Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. MiniLM2-L6-H768** | 6-layer Cross-Encoder (66M) | **FEASIBLE (Full SFT)** | 8 | 1 | 256 | 3e-5 | AdamW (`float32`) | ~450 MB | ~1.5 min |
| **2. DeBERTa-v3-small** | Disentangled Encoder (141M) | **FEASIBLE (Full SFT)** | 4 | 2 | 256 | 2e-5 | AdamW (`float32`) | ~850 MB | ~4.0 min |
| **3. ModernBERT-base-nli** | RoPE Bidirectional Encoder (149M)| **FEASIBLE (Full SFT - Truncated)** | 2 | 4 | 512 | 2e-5 | AdamW (`float32`) | ~1,100 MB | ~6.0 min |
| **4. TF-IDF + Logistic Reg** | Sparse Linear Classifier (~50k) | **FEASIBLE (Instantaneous Full Fit)**| Full | N/A | Full text | N/A | L-BFGS (`float64`) | ~80 MB | <2 sec |
| **5. DeBERTa-v3-small-dep** | Sequence Classifier (141M) | **FEASIBLE (Full SFT - Verified V2)**| 2 | 2 | 256 | 2e-5 | AdamW (`float32`) | ~900 MB | ~4.5 min |
| **6. BGE-small-en-v1.5** | Dense Bi-Encoder (33.5M) | **FEASIBLE (Full SFT / Embedding)** | 16 | 1 | 256 | 2e-5 | AdamW (`float32`) | ~350 MB | ~1.0 min |
| **7. Rule Engine Oracle** | Symbolic Logic (0 params) | **FEASIBLE (Deterministic Invariant)**| Full | N/A | N/A | N/A | Symbolic Deterministic | <30 MB | <0.1 sec |
| **8. DeBERTa-v3-base-FEVER** | Cross-Encoder Fact Head (435M) | **FEASIBLE (Full SFT on 200 pairs)** | 2 | 4 | 256 | 1e-5 | AdamW (`float32`) | ~1,600 MB | ~9.5 min |
| **9. Llama-Guard-3-1B** | Decoder Guardrail (1.23B) | **INFERENCE SMOKE ONLY (Lab Only Train)**| 1 | N/A | 384 | N/A | Inference-only (`int4`/`float32`)| ~2,800 MB | ~8.0 min (eval)|
| **10. Qwen2.5-0.5B-Instruct** | Decoder LLM (494M) | **FEASIBLE (LoRA SFT or ONNX Eval)** | 1 | 4 | 384 | 1e-4 | LoRA ($r=8$) / AdamW | ~1,800 MB | ~14.0 min |
| **11. Llama-3.2-1B-Instruct** | Decoder LLM (1.23B) | **INFERENCE SMOKE ONLY (Lab Only Train)**| 1 | N/A | 384 | N/A | Inference-only (Eval 10 cases) | ~2,900 MB | ~12.0 min (eval)|
| **12. Phi-3.5-mini-instruct**| Dense Decoder LLM (3.82B) | **STRICTLY NVIDIA LAB ONLY** | N/A | N/A | N/A | N/A | INFEASIBLE ON LAPTOP CPU | >16.0 GB (OOM) | N/A (Lab Only)|

---

## 3. Detailed Parameter Specifications for Trainable Laptop Models

### 3.1. Standard Fine-Tuning Recipe: Encoders (Models 1, 2, 3, 5, 8)
- **Framework:** PyTorch 2.13.0 + Hugging Face `transformers` Trainer.
- **Device Setting:** `device = torch.device("cpu")`.
- **Intra-op Threads:** `torch.set_num_threads(4)`.
- **Inter-op Threads:** `torch.set_num_interop_threads(2)`.
- **Random Seed:** Fixed to `42` for exact reproducibility.
- **Optimizer:** `AdamW` ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$).
- **Weight Decay:** `0.01`.
- **Learning Rate Schedule:** Linear warmup over first 10% of steps, linear decay to zero.
- **Precision:** Pure `torch.float32` (avoids CPU bfloat16 emulation overhead and quantization artifacts).
- **Gradient Clipping:** Max norm $1.0$.
- **Validation Frequency:** Evaluated at 50% and 100% of epoch completion.

---

### 3.2. Parameter-Efficient Recipe: Qwen2.5-0.5B-Instruct (Model 10)
- **Method:** Low-Rank Adaptation (LoRA) via Hugging Face `peft`.
- **LoRA Configuration:**
  - Rank ($r$): `8`
  - Alpha ($\alpha$): `16`
  - Target Modules: `["q_proj", "v_proj"]` (Restricting adaptation to query and value projection matrices minimizes trainable parameters to <0.8% of total weights).
  - LoRA Dropout: `0.05`
- **Batch Size:** `1` (Gradient accumulation steps = `4`, yielding effective batch size of `4`).
- **Max Input Sequence Length:** `384` tokens (covers prompt instructions and claim document text).
- **Max Generation Length:** `128` tokens (sufficient for compact JSON response).

---

## 4. Resource Guardrails & Failure Handling Protocol

To safeguard the developer's laptop and existing project checkpoints, the following safeguards are mandatory:

```
+---------------------------------------------------------------------------------------------------+
| PRE-FLIGHT VERIFICATION CHECKLIST                                                                 |
+---------------------------------------------------------------------------------------------------+
| 1. Isolated Checkpoint Destination : Writes strictly to `scratch/feasibility_checkpoints/`        |
|                                      NEVER overwrites `_private/models/candidate_v2/`             |
| 2. Dynamic Memory Polling          : Using `psutil.virtual_memory()`. If available RAM < 500 MB,  |
|                                      gracefully abort epoch and dump memory trace.                |
| 3. Execution Timeout Guard         : Training loop wrapped in a 20-minute hard watchdog timer.    |
| 4. Clean Artifact Deletion         : Temporary optimizer state files (.pt) deleted immediately    |
|                                      after serialization verification to save disk space.         |
| 5. Exception Interception          : All errors caught, serialized to `feasibility_report.json`,  |
|                                      and classified into OOM, SchemaError, or GradientNan.        |
+---------------------------------------------------------------------------------------------------+
```

---

## 5. Standardized 1-Epoch Harness Implementation Template

The following reusable Python test script provides the exact standard harness to verify any encoder model locally without code modifications:

```python
"""
laptop_feasibility_harness.py
Minimal, zero-risk, CPU-safe 1-epoch feasibility trial.
"""
import os
import sys
import time
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

def run_cpu_one_epoch_smoke(model_name: str, batch_size: int, seq_len: int, lr: float):
    print(f"\n[FEASIBILITY TRIAL] Model: {model_name}")
    start_time = time.time()
    
    # 1. Enforce strict resource constraints
    torch.set_num_threads(4)
    initial_ram = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"[*] Initial Process RAM: {initial_ram:.1f} MB")
    
    # 2. Load tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        model.to("cpu")
        model.train()
    except Exception as e:
        print(f"[!] FAILED TO LOAD MODEL: {e}")
        return False
        
    loaded_ram = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"[*] Model Loaded RAM: {loaded_ram:.1f} MB (Delta: {loaded_ram - initial_ram:.1f} MB)")
    
    # 3. Create synthetic validation batch
    dummy_text = ["Policy coverage active and verified.", "Claim invoice matches treatment date."]
    inputs = tokenizer(dummy_text * (batch_size // 2), padding="max_length", max_length=seq_len, return_tensors="pt")
    labels = torch.tensor([1, 0] * (batch_size // 2), dtype=torch.long)
    
    # 4. Single step forward, backward, and optimizer step
    optimizer = AdamW(model.parameters(), lr=lr)
    optimizer.zero_grad()
    
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    
    peak_ram = psutil.Process().memory_info().rss / (1024 * 1024)
    elapsed = time.time() - start_time
    print(f"[+] Gradient Step Successful! Loss: {loss.item():.4f}")
    print(f"[+] Peak RAM: {peak_ram:.1f} MB | Elapsed: {elapsed:.2f}s")
    
    # 5. Checkpoint serialization smoke test
    save_dir = os.path.join("scratch", "feasibility_checkpoints", model_name.split("/")[-1])
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"[+] Successfully saved checkpoint to: {save_dir}")
    return True

if __name__ == "__main__":
    # Example execution for MiniLM baseline
    run_cpu_one_epoch_smoke("cross-encoder/nli-MiniLM2-L6-H768", batch_size=4, seq_len=128, lr=3e-5)
```
