# Models, Baselines, and Benchmark Architecture

The `certificate-recovery` engine requires **no machine learning dependencies**. Optional AI/ML components serve advisory prediction, semantic verification, and multi-agent research benchmarking inside `research_benchmark/`.

## 12-Model Comparative Benchmark Registry

Models are evaluated strictly within four task-aligned groups rather than a single cross-task leaderboard:

| # | Group | Model / Checkpoint | Parameters | Role & Status |
|---|---|---|---|---|
| 1 | **Group 1: NLI & Semantic Reasoning** | `cross-encoder/nli-MiniLM2-L6-H768` (`b95119c`) | 66M | Fast fallback baseline; 5/6 on `baseline_tests.json` (fails `nli-003` at 99.57% contradiction) |
| 2 | **Group 1: NLI & Semantic Reasoning** | `cross-encoder/nli-deberta-v3-small` (`fa28048`) | 141M | Pinned primary NLI baseline & V2 backbone; 5/6 on `baseline_tests.json` (fails `nli-003` at 99.33% contradiction) |
| 3 | **Group 1: NLI & Semantic Reasoning** | `answerdotai/ModernBERT-base` | 149M | Long-context (8,192-token) policy NLI comparator (`research_benchmark/` scaffold) |
| 4 | **Group 2: Dependency Prediction** | `TF-IDF + Calibrated LogisticRegression` | ~50k sparse | Non-neural linear baseline in provenance suite |
| 5 | **Group 2: Dependency Prediction** | Custom `DeBERTa-v3-small` + 2-layer MLP (`Candidate V2`) | 141M | **Core fine-tuned subject model** ($\tau = 0.20$, saved privately; V3 root-explicit variant prepared/unrun) |
| 6 | **Group 2: Dependency Prediction** | `BAAI/bge-small-en-v1.5` | 33.5M | Dense bi-encoder cosine-similarity comparator |
| 7 | **Group 3: Evidence & Faults** | `distilbert/distilbert-base-uncased` | 66M | Compact trainable evidence classification baseline |
| 8 | **Group 3: Evidence & Faults** | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | 435M | FEVER fact-verification encoder for rationale grounding |
| 9 | **Group 3: Evidence & Faults** | `meta-llama/Llama-Guard-3-1B` | 1.23B | Policy/output guardrail classifier (GPU lab training / small inference) |
| 10 | **Group 4: Generative Specialists** | `Qwen/Qwen2.5-0.5B-Instruct` | 494M | Integrated compliance proposer (`model_agent.py`, INT8 ONNX local eval) |
| 11 | **Group 4: Generative Specialists** | `meta-llama/Llama-3.2-1B-Instruct` | 1.23B | 1B instruction-tuned specialist comparator |
| 12 | **Group 4: Generative Specialists** | `microsoft/Phi-3.5-mini-instruct` | 3.82B | GPU cluster frontier reasoning ceiling (`DEFERRED_TO_NVIDIA`) |
| **Ref** | **Authoritative Symbolic Oracle** | `MandatoryRuleGate` + `RuleAgent` (Arm F) | 0 | Deterministic ground-truth safety baseline (0 unsafe approvals across ablation suites) |

Full structured metadata is maintained canonically in `research_benchmark/registry.py`.

## Pinned Semantic Baselines and Remote Clients

- **Optional Remote Client (`remote_model_client.py`, `run_baseline.py`):** Connects to public Hugging Face Gradio Spaces (`nli-zero-gpu` via `MridulSharma02/contradiction-detector`, `nli-cpu` via `tasksource/ModernBERT-zero-shot-nli`). Both remote endpoints serve alternative models rather than attesting `cross-encoder/nli-deberta-v3-small` weights. Credentials use `.env` (never committed) and are redacted from `results/baseline_*.json`.
- **Immutable Pins (`config/pinned_smoke_models.json`):** Locks exact 40-character revisions and label order (`["contradiction", "entailment", "neutral"]`) for direct offline/Colab smoke comparisons without the Space NLTK heuristic filter (`notebooks/02_baseline_model.ipynb`).
- **Semantic Advice Containment (`semantic_advice.py`):** Captures `(artifact_id, version, sha256)` bindings on prepare, verifies request correlation and freshness under the engine lock, and records `ACCEPTED`, `IGNORED`, or `REJECTED_STALE` without modifying dependency edges or certificates.

## Candidate V2 Read-Only Inference & Qwen2.5 Compliance Adapter

- **Candidate V2 Loader (`src/cert_recovery/v2_inference.py`):** Verifies saved `manifest.json`, `protocol.json`, `config.json`, `selection.json`, and all SHA-256 digests under `checkpoints/best/` before running local CPU inference (`local_files_only=True`, `trust_remote_code=False`).
- **Qwen2.5-0.5B Prompt Calibration:** Evaluated locally across 4 compliance scenarios comparing Prompt v1 (0-shot) vs. Prompt v2 (2-shot JSON schema). Under 2-shot prompting, valid JSON and evidence citation rates reached 4/4 (100%) and verifier acceptance rose from 0/4 to 3/4, at ~37s CPU latency per call. Multi-agent benchmark evaluations in `results/model_agent/model_evaluation_results.json` used `mock_compliance_backend` (`live_backend_executed: false`).

## Execution Tiers (CPU Feasibility vs. GPU Cluster)

Hyperparameters are centralized in `config/research_config.yaml`:

- **CPU Feasibility Profile:** 4 PyTorch threads, `float32`, memory ceiling guard (`<3.5 GB` RSS via `research_benchmark/hardware.py`), 1-epoch smoke verification for Groups 1–3 (`bs=2..8`) and Qwen2.5 LoRA (`r=8`). Multi-billion-parameter LLMs (`Llama-Guard-3-1B`, `Llama-3.2-1B`, `Phi-3.5-mini`) are deferred from local training to GPU nodes.
- **GPU Cluster Profile (Planned / Scripts Ready):** Multi-seed runs (`[42, 101, 2024]`), `bf16` mixed precision, sequence lengths up to 8,192 for ModernBERT and 1,024–2,048 for LoRA/QLoRA (`r=16..32`), with blind held-out test sets frozen prior to training.
