# NVIDIA Laboratory Extensive Training Roadmap
**Project:** Verifiable Multi-Agent Fault-Tolerant Adjudication System with Cryptographic Causal Provenance  
**Local Workspace:** `certificate-recovery`  
**Target Hardware:** University Laboratory NVIDIA GPU Clusters (A100 / H100 / RTX 4090)  
**Author:** Senior Deep Learning Engineer & MLOps Specialist  

---

## 1. Laboratory Overview & Hardware Topology

Following local 1-epoch feasibility verification on the laptop, extensive training and rigorous comparative benchmarking will be conducted in the university laboratory. Because university server availability fluctuates, this roadmap defines standardized execution profiles for **three distinct hardware tiers**:

```
+---------------------------------------------------------------------------------------------------+
| TIER 1: Enterprise Research Node (Preferred)                                                      |
| - Hardware : 1x or 2x NVIDIA A100-SXM4 (40GB / 80GB) or H100 (80GB)                               |
| - Precision: Native Brain Floating Point (`bf16`) with TensorFloat-32 (TF32)                      |
| - Focus    : ModernBERT (full 8k context), LoRA for Phi-3.5-mini (3.82B) and Llama-3.2 (1.23B)    |
+---------------------------------------------------------------------------------------------------+
| TIER 2: High-End Workstation Node (Standard Lab Server)                                           |
| - Hardware : 1x NVIDIA GeForce RTX 4090 / 3090 (24GB VRAM) or RTX 6000 Ada                      |
| - Precision: `bf16` or `fp16` with FlashAttention-2 enabled                                       |
| - Focus    : Full fine-tuning of all Encoders (Groups 1-3), LoRA SFT for Qwen2.5 & Llama-3.2      |
+---------------------------------------------------------------------------------------------------+
| TIER 3: Constrained Shared GPU Node (Fallback Lab Server)                                         |
| - Hardware : 1x NVIDIA RTX 3060 / 4060 / T4 (8GB - 12GB VRAM)                                     |
| - Precision: `fp16` mixed precision; 4-bit NormalFloat (QLoRA) for generative models              |
| - Focus    : Encoder fine-tuning; QLoRA for Group 4 LLMs                                          |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Multi-Tier Training Configurations & Hyperparameters

| Research Group & Model | Parameters | Target Hardware | Effective Batch Size | Max Seq Len | Learning Rate | Optimizer & Precision | Epochs & Early Stopping | Checkpoint Retention |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Group 1: MiniLM-L6** | 66M | Tier 1 / 2 / 3 | 32 (32 × 1) | 512 | 3e-5 | AdamW (`bf16` / `fp16`) | 10 epochs (Patience = 3 on Val F1) | Best 2 checkpoints |
| **Group 1: DeBERTa-v3-small** | 141M | Tier 1 / 2 / 3 | 16 (16 × 1) | 512 | 2e-5 | AdamW (`bf16` / `fp16`) | 8 epochs (Patience = 2 on Val F1) | Best 2 checkpoints |
| **Group 1: ModernBERT-base** | 149M | Tier 1 / 2 | 8 (8 × 1) | 4,096–8,192 | 2e-5 | AdamW (`bf16` FlashAttn-2) | 6 epochs (Patience = 2 on Val Loss)| Best 2 checkpoints |
| **Group 2: DeBERTa-v3-dep** | 141M | Tier 1 / 2 / 3 | 16 (8 × 2) | 256 | 2e-5 | AdamW (`bf16` / `fp16`) | 10 epochs (Patience = 3 on Val F1) | Best 2 checkpoints |
| **Group 2: BGE-small-en** | 33.5M | Tier 1 / 2 / 3 | 64 (64 × 1) | 256 | 3e-5 | AdamW (`bf16` / `fp16`) | 5 epochs (Cosine decay) | Best 2 checkpoints |
| **Group 3: DistilBERT-base** | 66M | Tier 1 / 2 / 3 | 32 (32 × 1) | 512 | 2e-5 | AdamW (`bf16` / `fp16`) | 8 epochs (Patience = 2 on Val F1) | Best 2 checkpoints |
| **Group 3: DeBERTa-FEVER** | 435M | Tier 1 / 2 | 16 (8 × 2) | 512 | 1.5e-5 | AdamW (`bf16` / `fp16`) | 8 epochs (Patience = 2 on Val F1) | Best 2 checkpoints |
| **Group 3: Llama-Guard-3-1B** | 1.23B | Tier 1 / 2 / 3 | 8 (2 × 4) | 1,024 | 1e-4 | LoRA ($r=16$) (`bf16`/QLoRA)| 5 epochs (Patience = 2) | Best 1 checkpoint |
| **Group 4: Qwen2.5-0.5B** | 494M | Tier 1 / 2 / 3 | 16 (4 × 4) | 1,024 | 2e-4 | LoRA ($r=16$) (`bf16` / `fp16`) | 5 epochs (Patience = 2) | Best 2 checkpoints |
| **Group 4: Llama-3.2-1B** | 1.23B | Tier 1 / 2 / 3 | 8 (2 × 4) | 2,048 | 1e-4 | LoRA ($r=16$) (`bf16`/QLoRA)| 5 epochs (Patience = 2) | Best 2 checkpoints |
| **Group 4: Phi-3.5-mini** | 3.82B | Tier 1 / 2 | 4 (1 × 4) | 2,048 | 5e-5 | QLoRA ($r=32$) (`bf16`) | 4 epochs (Patience = 1) | Best 1 checkpoint |

---

## 3. Optuna Bayesian Optimization Protocol

To eliminate arbitrary manual hyperparameter tuning, we execute a **30-trial Optuna Bayesian search** using the Tree-structured Parzen Estimator (TPE):

```
Target Objective: Maximize Validation Macro-F1 (or Minimize Expected Calibration Error)
Search Hyperparameter Space:
- Learning Rate     : Log-uniform distribution [5e-6, 3e-4]
- Warmup Ratio      : Uniform distribution [0.03, 0.15]
- Weight Decay      : Log-uniform distribution [1e-4, 1e-1]
- Effective Batch   : Categorical choice [8, 16, 32]
- LoRA Rank (LLMs)  : Categorical choice [8, 16, 32]
- LoRA Alpha (LLMs) : Fixed at 2 * LoRA Rank
```

### Triplicate Multi-Seed Verification:
Once the optimal hyperparameter tuple is identified on validation splits, each model is trained across **three independent seeds** (`42`, `101`, `2024`). Results on the final sealed test set are reported as $\text{Mean} \pm \text{Standard Deviation}$ to verify that performance gains are statistically robust and not seed anomalies.

---

## 4. Execution Dependency Matrix: Parallel vs. Sequential Jobs

```
+---------------------------------------------------------------------------------------------------+
| STAGE 1: PARALLEL ENCODER TRAINING (Can run concurrently across separate GPUs)                     |
| - Group 1: MiniLM, DeBERTa-v3-small, ModernBERT                                                   |
| - Group 2: Custom DeBERTa-v3, BGE-small bi-encoder                                                 |
| - Group 3: DistilBERT, DeBERTa-base-FEVER                                                          |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v (Validation Complete & Hyperparameters Frozen)
+---------------------------------------------------------------------------------------------------+
| STAGE 2: SEQUENTIAL GENERATIVE LLM ADAPTATION (Dedicated VRAM allocation)                          |
| - Group 4: Qwen2.5-0.5B LoRA SFT                                                                  |
| - Group 4: Llama-3.2-1B LoRA SFT                                                                  |
| - Group 4: Phi-3.5-mini QLoRA SFT                                                                 |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v (All Weights Frozen & Artifacts Hashed)
+---------------------------------------------------------------------------------------------------+
| STAGE 3: BLIND EVALUATION ON SEALED HELD-OUT TEST SETS (Exactly Once)                             |
| - Zero test-set leakage guaranteed by hash-partitioning and irreversible evaluation protocol       |
+---------------------------------------------------------------------------------------------------+
```

---

## 5. MLOps Experiment Tracking & Artifact Lifecycle

- **Experiment Tracking:** Logged via local MLflow instance (`_private/mlflow/`) and Weights & Biases (W&B).
- **Artifacts Recorded:** Raw training and validation loss curves, validation macro-F1 trajectories, confusion matrices, ROC curves, precision-recall curves, and hardware telemetry (peak GPU VRAM, power draw, throughput in samples/second).
- **Checkpoint Retention:** Keep only the top-2 best checkpoints based on validation macro-F1 to prevent disk exhaustion.
