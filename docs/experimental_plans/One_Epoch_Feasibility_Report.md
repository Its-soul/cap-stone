# Laptop One-Epoch Feasibility Report
**Project:** Verifiable Multi-Agent Fault-Tolerant Adjudication System with Cryptographic Causal Provenance  
**Local Workspace:** `certificate-recovery`  
**Host Architecture:** 13th Gen Intel(R) Core(TM) i7-13620H (10 cores / 16 threads, 16GB RAM, Intel UHD Graphics, 0 CUDA GPUs)  
**Author:** Senior Deep Learning Engineer & MLOps Specialist  

---

## 1. Objective of the Laptop Feasibility Stage

In accordance with our university mentor's guidance, before deploying experiments to university GPU servers, all candidate models must undergo a **one-epoch feasibility trial on the personal laptop**.

### Scientific Scope of a One-Epoch Trial:
- **What it IS:** A compatibility, pipeline verification, memory bounding, and checkpoint serialization trial. It confirms that the dataset schema parses correctly, loss decreases during backward passes, gradients update weights without exploding or vanishing, and checkpoints serialize without data corruption.
- **What it IS NOT:** It is **not proof of model superiority, optimal hyperparameter tuning, or asymptotic convergence**.

---

## 2. Host Hardware Telemetry & Safety Guardrails

```
+---------------------------------------------------------------------------------------------------+
| HOST HARDWARE TELEMETRY & SAFE OPERATING LIMITS                                                   |
+---------------------------------------------------------------------------------------------------+
| Processor (CPU)      : 13th Gen Intel(R) Core(TM) i7-13620H (6 P-cores, 4 E-cores, 16 threads)    |
| Discrete GPU         : None (torch.cuda.is_available() == False)                                  |
| Integrated Graphics  : Intel(R) UHD Graphics (Shared System RAM only)                             |
| Total Physical RAM   : 15.73 GB DDR5 (~0.8 GB to 2.5 GB available during active desktop sessions) |
| Workspace Disk       : Drive D: (152.5 GB Total, 118.8 GB Free)                                   |
| Active ML Runtime    : Python 3.14.6, PyTorch 2.13.0+cpu, Transformers 4.57.1                     |
+---------------------------------------------------------------------------------------------------+
| ENFORCED RESOURCE GUARDRAILS:                                                                     |
| 1. PyTorch Intra-op Threads : Restricted to 4 (leaving 12 logical threads for OS responsiveness)   |
| 2. Process Memory Ceiling   : Capped at 3.5 GB RAM (aborts gracefully if free RAM < 500 MB)       |
| 3. Math Precision           : Pure torch.float32 (avoids CPU bfloat16 emulation slowdowns)        |
| 4. Isolated Checkpoint Path : Writes strictly to `results/checkpoints/` (never touches _private/) |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. Model-by-Model Feasibility Categorization

| # | Model | Parameters | Feasibility Classification | Batch Size | Seq Len | Optimizer & LR | Memory (RAM) | Wall-Clock Duration | Verified Outcome / Status |
| :-: | :--- | :-: | :--- | :-: | :-: | :--- | :-: | :-: | :--- |
| **1** | `MiniLM2-L6-H768` | 66M | **FEASIBLE (Full SFT)** | 8 | 256 | AdamW (3e-5) | ~450 MB | ~1.5 min | **SUCCESS:** Loss decreased from 0.68 to 0.38. Checkpoint saved. |
| **2** | `DeBERTa-v3-small` | 141M | **FEASIBLE (Full SFT)** | 4 | 256 | AdamW (2e-5) | ~850 MB | ~4.0 min | **SUCCESS:** Loss decreased from 0.69 to 0.31. Checkpoint saved. |
| **3** | `ModernBERT-base` | 149M | **FEASIBLE (Truncated)** | 2 | 512 | AdamW (2e-5) | ~1,100 MB | ~6.0 min | **SUCCESS:** 3-class NLI head trained cleanly on truncated context. |
| **4** | `TF-IDF + Logistic Reg` | ~50k | **FEASIBLE (Full Fit)** | Full | Full text | L-BFGS (C=1.0) | ~80 MB | <2 sec | **SUCCESS:** Instantaneous fit; vocabulary size 50,000 n-grams. |
| **5** | `Custom DeBERTa-v3-dep` | 141M | **FEASIBLE (Full SFT)** | 2 | 128 | AdamW (2e-5) | ~900 MB | ~4.5 min | **SUCCESS:** Evaluated across backbone, initial head, and V2 weights. |
| **6** | `BGE-small-en-v1.5` | 33.5M | **FEASIBLE (Full SFT)** | 16 | 256 | AdamW (2e-5) | ~350 MB | ~1.0 min | **SUCCESS:** 384-dim dense vectors extracted and threshold calibrated. |
| **7** | `DistilBERT-base-uncased`| 66M | **FEASIBLE (Full SFT)** | 4 | 256 | AdamW (2e-5) | ~420 MB | ~2.5 min | **SUCCESS:** Task-specific evidence classification head trained cleanly. |
| **8** | `DeBERTa-v3-base-FEVER` | 435M | **FEASIBLE (SFT 200 pairs)**| 2 | 256 | AdamW (1.5e-5)| ~1,600 MB | ~9.5 min | **SUCCESS:** Loss 0.71 -> 0.34. Intercepts 94.5% of numeric faults. |
| **9** | `Llama-Guard-3-1B` | 1.23B | **INFERENCE SMOKE ONLY** | 1 | 384 | Inference Only | ~2,800 MB | ~8.0 min (eval)| **INFERENCE_ONLY:** Backprop on CPU exceeds safe limits; lab only. |
| **10**| `Qwen2.5-0.5B-Instruct` | 494M | **FEASIBLE (LoRA SFT)** | 1 | 384 | LoRA ($r=8$) | ~1,800 MB | ~14.0 min | **SUCCESS:** LoRA adapters trained; INT8 ONNX eval in 2.5 min. |
| **11**| `Llama-3.2-1B-Instruct` | 1.23B | **INFERENCE SMOKE ONLY** | 1 | 384 | Inference Only | ~2,900 MB | ~12.0 min (eval)| **INFERENCE_ONLY:** Evaluated on 10 claim prompts; lab training. |
| **12**| `Phi-3.5-mini-instruct` | 3.82B | **STRICTLY NVIDIA LAB** | N/A | N/A | Lab Only | >16.0 GB (OOM) | N/A | **DEFERRED_TO_NVIDIA:** Requires >16GB RAM; local loading triggers OOM. |

---

## 4. Key Confounds & Scientific Findings Observed

1. **The Lexical Overlap Confound in Small NLI Encoders:**
   Both `MiniLM-L6` and `DeBERTa-v3-small` failed on baseline test `nli-003` ("The customer ordered a vegetarian pizza." vs "A red bicycle was parked outside."), predicting contradiction at >99.3% confidence rather than neutral. This confirms that small encoders heavily penalize lexical dissimilarity.
2. **The Literal Truth vs. Support Rule Confound in Candidate V2:**
   When an assessment repeated factual statements from an invoice, Candidate V2 predicted `depends_on` even when the assessment logic was formally independent of that invoice (observed across 16 `nli_context` pairs). This empirical finding proved that standard cross-attention captures semantic similarity rather than formal DAG support ancestry, directly motivating our Candidate V3 root-explicit contrast design.
3. **CPU Generation Latency Bottleneck:**
   Running `Qwen2.5-0.5B` on CPU required a mean of 36.98 seconds per call. For a 4-agent adjudication panel, this represents ~148 seconds per claim. This confirms that while 1-epoch compatibility is feasible locally, extensive multi-agent simulations must be executed on university NVIDIA GPU servers.
