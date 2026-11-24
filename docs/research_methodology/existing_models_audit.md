# Existing Models Audit Report
**Project:** Verifiable Multi-Agent Fault-Tolerant Adjudication System with Cryptographic Causal Provenance  
**Repository:** [https://github.com/Its-soul/cap-stone](https://github.com/Its-soul/cap-stone)  
**Investigation Date:** October 2026  
**Auditor:** Senior AI/ML Research Scientist & Deep Learning Specialist  

---

## 1. Executive Summary & Architecture Decomposition

An exhaustive audit of the repository (`d:/Code/Github/certificate-recovery`) was conducted across all production modules (`src/cert_recovery/`), test suites (`tests/`), scripts, configuration files, and empirical research archives (`_private/runs/`, `_private/notebooks/Research_Evaluation.ipynb`).

The primary architectural finding is that the project implements a **hybrid architecture** with a strict, non-negotiable boundary between **untrusted advisory AI/ML models** and **authoritative deterministic cryptographic adjudication mechanisms**:

```
+---------------------------------------------------------------------------------------------------+
|                                   UNTRUSTED ADVISORY ML LAYER                                     |
|  - Candidate V2/V3 Dependency Classifier (DeBERTa-v3-small)                                        |
|  - Semantic Advice & Cross-Encoder Baselines (MiniLM-L6, DeBERTa-v3-small)                        |
|  - Specialist Agent LLM Proposers (Qwen2.5-0.5B-Instruct ONNX)                                    |
|  - Output: Advisory edge predictions, NLI semantic scores, structured JSON claim assessments       |
+--------------------------------------------------+------------------------------------------------+
                                                   |
                                                   v (Untrusted Proposals: 0 Decisional Authority)
+--------------------------------------------------+------------------------------------------------+
|                              AUTHORITATIVE DETERMINISTIC LAYER                                    |
|  - MandatoryRuleGate (Policy exclusion rules: coverage, consent, documents, fraud)               |
|  - DeterministicValidator (Schema enforcement, domain boundary invariants)                       |
|  - CertificateSystem (Cryptographic Directed Acyclic Graph - DAG, state machine lifecycle)        |
|  - EventDAG & Ledger (SHA-256 parent hash chaining, tip commitment, external anchor verification) |
|  - Selective Recovery Engine (Causal invalidation, lineage recomputation, valid state reuse)      |
+---------------------------------------------------------------------------------------------------+
```

### Key Architectural Tenet
**No machine learning prediction can issue a certificate, override a deterministic exclusion rule, or establish an authoritative dependency edge.** ML components provide advisory suggestions (e.g., highlighting likely document dependencies, suggesting claim assessment drafts, or flagging semantic contradictions). If an ML model fails, hallucination is intercepted by schema validation, policy rules, or consensus thresholds, resulting in a safe transition to human `REVIEW` rather than an erroneous `APPROVE`.

---

## 2. Comprehensive Audit of Existing AI/ML Models

Below is the verified audit of every model checkpoint, architecture, and pipeline identified in the repository code and experimental archives.

| Model Checkpoint | Architecture | Publication Year | Release Year | Project Role | Training / Eval Dataset | Implementation Status | Verified Empirical Results | Known Limitations & Confounds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`cross-encoder/nli-MiniLM2-L6-H768`** (rev `b95119c`) | 6-layer BERT-style Transformer Cross-Encoder ($H=768$, 12 heads) | **2020** (MiniLMv2 paper, Wang et al.) | **2021** (HF checkpoint release) | Semantic consistency check; NLI baseline comparator | SNLI + MultiNLI (Pretrained); evaluated on 6 authored baseline pairs (`baseline_tests.json`) | **Implemented & Tested** (Inference-only smoke baseline) | 5/6 (83.3%) match on baseline tests. Predicted contradiction on test `nli-003` (pizza vs bicycle) at 99.57% (expected neutral). | 1. Severe lexical mismatch bias (treats unrelated sentences as contradictions rather than neutral).<br>2. Cannot infer computational or causal dependency from semantic similarity.<br>3. Prone to shortcut learning on lexical overlap. |
| **`cross-encoder/nli-deberta-v3-small`** (rev `fa28048`) | Disentangled attention Transformer Cross-Encoder (ELECTRA-style pretraining) | **2021** (DeBERTa-v3 paper, He et al., ICLR 2023) | **2021 / 2022** (HF checkpoint release) | Primary baseline for semantic advice; cross-encoder NLI scoring (`semantic_advice.py`) | MNLI + SNLI (Pretrained); evaluated on 6 authored baseline pairs | **Implemented & Tested** (Pinned primary baseline) | 5/6 (83.3%) match on baseline tests. Predicted contradiction on test `nli-003` at 99.33% (expected neutral). | 1. Exhibits the same neutral-vs-contradiction ambiguity on unrelated concepts.<br>2. Higher inference latency than MiniLM (~3x slower per pair on CPU).<br>3. Semantic entailment does not imply causal or computational dependency. |
| **`Candidate V2 Dependency Classifier`** (`deberta-v3-small` + Binary Head) | Sequence Classification Encoder + 2-layer MLP head (`independent` [0] vs `depends_on` [1]) | **2021** (DeBERTaV3 backbone); **2026** (Head fine-tuning) | **2026** (Trained checkpoint in `_private/models/candidate_v2/`) | Binary prediction of declared computational dependency between source artifacts and assessments | Synthetic controlled provenance dataset (`controlled_data.py`: 256 rows train, 64 val, 128 stress, 120 final) | **Fine-Tuned & Evaluated** (Trained on CPU, 504.6s, AdamW, lr=2e-5, bs=2) | **Val macro-F1:** 0.873 (acc 87.5% at threshold 0.2).<br>**Stress acc:** 82.8% (42 improvements, 49 regressions).<br>**Final held-out acc:** 91.7% (macro-F1 0.916). | **Major Confound Identified:** Failed on 16 `nli_context` test pairs where hypothesis repeated facts from candidate artifact. The model confounded literal semantic truth with declared artifact support rules, revealing that standard cross-attention captures lexical overlap rather than causal lineage. |
| **`Candidate V3 Root-Explicit Contrast`** (`deberta-v3-small` planned) | Sequence Classification Encoder with root-explicit contrastive conditioning | **2021** (Backbone); **2026** (Dataset proposal) | **2026** (Proposed in `docs/DEPENDENCY_TARGET.md` & `v3_preparation.py`) | Causal dependency edge classifier with explicit ancestral closure | Proposed 512 root-explicit contrast rows (256 train, 64 val, 128 future-final) with explicit negative rule: *"All sources have no supports."* | **PREPARED_NOT_RUN** (Dataset generated and sealed, training deferred) | No training executed. Dataset sealed to prevent data leakage and benchmark contamination. | 1. Relies on synthetic prompt-engineered contrast pairs.<br>2. Requires empirical validation of whether explicit negative ancestral rules prevent lexical shortcutting. |
| **`onnx-community/Qwen2.5-0.5B-Instruct`** (rev `22942cb`, `model_int8.onnx`) | 24-layer Transformer Decoder with RoPE, RMSNorm, SwiGLU, tied embeddings (INT8 ONNX) | **2024** (Qwen2.5 technical report, Alibaba) | **2024** (HF checkpoint release) | Generative specialist agent proposer (compliance, risk, documents); generates structured JSON decisions | Evaluated on 4 representative insurance claim scenarios across Prompt v1 (0-shot) and Prompt v2 (2-shot) | **Implemented & Evaluated** (Real CPU ONNX inference via `model_agent.py` and `Research_Evaluation.ipynb`) | **Prompt v1 (0-shot):** 1/4 (25%) raw correct, 0/4 verifier accepted.<br>**Prompt v2 (2-shot):** 3/4 (75%) raw correct, 4/4 (100%) valid JSON, 4/4 (100%) evidence fidelity, 3/4 (75%) verifier accepted.<br>**CPU Latency:** Load 8.15s, generation mean 36.98s/call. | 1. High CPU generation latency (~37s per call on Intel i7) makes multi-agent iterative adjudication slow on local hardware.<br>2. 0-shot prompt failure highlights sensitivity to prompt structure.<br>3. Occasional semantic drift in complex multi-clause policy reasoning. |
| **`Remote Model Worker Spaces`** (Hugging Face Spaces Client) | Configurable remote inference client (`remote_model_client.py`, `scripts/remote_model_worker.py`) | **2024–2026** | **2026** | Remote GPU offloading for DeBERTa / ModernBERT and LLM worker tasks | Remote endpoints (`MridulSharma02/cap-stone-model-space`, `tasksource/ModernBERT-base-nli`) | **Implemented Client Interface** (Awaiting remote lab worker deployment) | Client handshake, payload serialization, and fallback mechanisms verified in unit tests. | Remote network latency and dependency on external API availability; requires authenticated tokens. |

---

## 3. Dissecting the Architectural Core: Deterministic vs. Learned

A critical requirement for this research project is distinguishing what is **genuinely learned by ML** from what is **computed deterministically**. The table below establishes this boundary across all system modules:

| Component | Implementation File | Underlying Mechanism | Decisional Authority | Research Function |
| :--- | :--- | :--- | :--- | :--- |
| **`CertificateSystem`** | `src/cert_recovery/engine.py` | Directed Acyclic Graph (DAG) state machine; strict transition checks (`PENDING -> VERIFIED -> SUPERSEDED / REVOKED`) | **Authoritative** (Sole issuer of valid adjudication state) | Cryptographic verification and selective recovery. |
| **`EventDAG` & `EventLedger`** | `src/cert_recovery/ledger.py` | SHA-256 parent hash chaining, Merkle-style root commitment, external anchor verification | **Authoritative** (Tamper-evident log) | Guarantees non-repudiation and causal ordering. |
| **`MandatoryRuleGate`** | `src/cert_recovery/consensus.py` | Hard-coded Boolean policy predicate evaluation (`active_policy == True`, `coverage_valid == True`, `consent == True`) | **Authoritative** (Unconditional veto) | Overrides all agent consensus and ML predictions. If a mandatory exclusion fails, claim is rejected immediately. |
| **`RuleAgent`** | `src/cert_recovery/domain.py` | Deterministic specialist evaluators for coverage, documents, provider risk, and identity | **Deterministic Baseline** | Provides 100% reproducible ground-truth decisions for benchmark evaluation. |
| **`Candidate V2 Classifier`** | `_private/models/candidate_v2/` | Fine-tuned Transformer Sequence Classifier (`deberta-v3-small`) | **Advisory Only** (Weight = 0 in final certificate issuance) | Recommends potential dependency edges between updated invoices and dependent assessments. |
| **`Qwen2.5-0.5B Specialist Agent`** | `src/cert_recovery/model_agent.py` | INT8 ONNX generative LLM generating structured JSON proposals | **Advisory Only** (Requires validation by `DeterministicValidator`) | Proposes domain assessments from unstructured textual claims. |

---

## 4. Empirical Evaluation of Past Experiments & Notebook Results

The notebook `_private/notebooks/Research_Evaluation.ipynb` and run directory `_private/runs/` document two comprehensive experimental evaluations:

### 4.1. Factorial Ablation Study (150 Adjudication Runs)
Conducted across 9 experimental arms evaluating permutations of Rule Gating, Specialist Composition, Confidence Weighting, and Fallback Recovery across 3 random seeds (42, 101, 2024):

```
+----------------------------------------------------------------------------------------------------+
| Arm | Rule Gate | Specialist Composition | Confidence Weight | Mean Acc | Unsafe Approvals | Notes  |
+-----+-----------+------------------------+-------------------+----------+------------------+--------+
|  A  |    No     |           No           |        No         |  51.8%   |       109        | Toxic  |
|  B  |    No     |          Yes           |        No         |  51.8%   |       109        | Toxic  |
|  C  |    No     |           No           |        Yes        |  50.5%   |       109        | Toxic  |
|  D  |    No     |          Yes           |        Yes        |  51.8%   |       109        | Toxic  |
|  E  |    Yes    |           No           |        No         |  98.7%   |         0        | Safe   |
|  F  |    Yes    |          Yes           |        No         | 100.0%   |         0        | Safe   |
|  G  |    Yes    |           No           |        Yes        | 100.0%   |         0        | Safe   |
|  H  |    Yes    |          Yes           |        Yes        | 100.0%   |         0        | Safe   |
|  I  |    Yes    |          Yes           |     Fallback      | 100.0%   |         0        | Safe   |
+----------------------------------------------------------------------------------------------------+
```

#### Key Scientific Findings from the Ablation Study:
1. **Rule Gate is Non-Negotiable (+48.2% Accuracy Jump):** Disabling the Rule Gate (Arms A–D) produces catastrophic failure, resulting in 109 unsafe claim approvals (approving claims where coverage had expired or fraud was flagged).
2. **Confidence Weighting Cannot Substitute for Rules:** Adding confidence weighting without a rule gate (Arm C vs Arm A) actually decreased accuracy from 51.8% to 50.5%. Highly confident incorrect model outputs overpowered correct low-confidence agents.
3. **Specialist Composition Solves Cross-Domain Voting Failures:** When specialist composition was disabled (Arm E), 42 false reviews occurred because non-policy specialist agents (e.g., document-check passing) outvoted single-policy exclusions before reaching final adjudication. Composition ensures each domain retains veto power in its respective sphere.

---

### 4.2. Repaired Selective Recovery Benchmarking
The selective recovery mechanism was benchmarked against single-pass full recomputation across varying claim batch sizes:

1. **Selective Invalidation Accuracy:** 100% of invalidated downstream certificates were correctly identified via BFS traversal of parent SHA-256 dependency edges.
2. **State Preservation:** In single-agent failure scenarios (e.g., invoice format corruption), 3 out of 4 specialist certificates from version $v_1$ were preserved without recomputation.
3. **Recomputation Efficiency:**
   - Full recomputation executes 4 agent evaluations per case.
   - Selective recovery executes only the invalidated agent (saving 3 agent calls).
   - In fast mock simulations, wall-clock speedup was modest (1.12x–1.27x) due to the relative overhead of SHA-256 DAG traversal versus instant mock execution.
   - However, when measured against the real `Qwen2.5-0.5B` ONNX model (which requires ~37s per agent call), selective recovery eliminates ~110 seconds of redundant CPU compute per invalidated claim.

---

## 5. Identified Limitations & Confounds in Existing Models

1. **The NLI vs. Causal Dependency Confound:**
   Cross-encoders trained on standard NLI datasets (MNLI, SNLI) assess *semantic entailment* ($P(\text{Hypothesis is True} \mid \text{Premise})$). They do not assess *computational dependency* ($P(\text{Output depends on Input Artifact})$). In Candidate V2, when an assessment repeated facts from an invoice, the NLI head flagged it as "entailed", even when the assessment logic was formally independent of that invoice.
2. **Context Leakage in Synthetic Training Data:**
   The Candidate V2 training set (`controlled_data.py`) contained template artifacts with high surface n-gram overlap between candidate artifacts and hypothesis statements. The model learned to exploit token matching shortcuts rather than tracing provenance.
3. **CPU Latency Bottleneck for Generative Specialist Agents:**
   The INT8 Qwen2.5-0.5B model takes 36.98 seconds per inference call on the laptop CPU. Running a 4-agent adjudication panel requires ~148 seconds per claim. This confirms that local laptop execution must focus on lightweight 1-epoch compatibility checks, while multi-agent iterative evaluations must be offloaded to the university NVIDIA GPU laboratory.
