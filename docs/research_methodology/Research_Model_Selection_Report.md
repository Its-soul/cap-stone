# Research Model Selection Report: 12-Model Comparative Framework
**Project:** Verifiable Multi-Agent Fault-Tolerant Adjudication System with Cryptographic Causal Provenance  
**Academic Degree:** Final-Year B.Tech CSE (AI/ML)  
**Author:** Senior AI/ML Research Scientist & Deep Learning Specialist  
**Local Workspace:** `certificate-recovery`  

---

## 1. Executive Summary & Experimental Philosophy

Automated claim adjudication systems that rely on monolithic large language models suffer from two foundational flaws:
1. **Unbounded Hallucination & Decisional Drift:** Models invent policy clauses or overlook explicit exclusions, generating costly erroneous approvals.
2. **Computational Waste & Lineage Opacity:** When an evidence document (such as a medical invoice or police report) changes, monolithic systems either recompute all reasoning from scratch or blindly reuse stale outputs.

This capstone project establishes a **hybrid verifiable adjudication architecture**:
- **Authoritative Deterministic Layer:** Enforces non-negotiable policy exclusion rules (`MandatoryRuleGate`), manages cryptographic certificates across a Directed Acyclic Graph (`CertificateSystem`), chains tamper-evident event hashes (`EventDAG`), and executes selective recovery following evidence updates.
- **Untrusted Advisory AI/ML Layer:** Specialized machine learning models that generate initial claim assessment proposals, screen for semantic contradictions, verify evidence citations, and predict candidate computational dependency edges.

To rigorously benchmark the advisory AI/ML layer, we have established an **experimental framework comprising exactly 12 models partitioned into four distinct research groups**. 

```
+---------------------------------------------------------------------------------------------------------+
|                                    FOUR COMPARATIVE EXPERIMENTAL GROUPS                                 |
+---------------------------------------------------------------------------------------------------------+
| Group 1: Semantic Reasoning & NLI              | Group 2: Computational Dependency Prediction           |
| (3-class Cross-Encoder Sequence Pair Classif.) | (Binary Edge Classification on Declared Provenance)    |
| - MiniLM2-L6-H768 (Compact Baseline)           | - TF-IDF + Logistic Regression (Linear Baseline)       |
| - DeBERTa-v3-small (Pinned Primary Baseline)   | - CUSTOM DeBERTa-v3-small (Core Subject Model)         |
| - ModernBERT-base (2024 SOTA 8k Context)       | - BGE-small-en-v1.5 (Dense Bi-Encoder Baseline)        |
+------------------------------------------------+--------------------------------------------------------+
| Group 3: Evidence & Fault Detection            | Group 4: Specialist Assessment Generation              |
| (Unsupported Claim & Guardrail Classification) | (Structured JSON Generation & Verifier Acceptance)     |
| - DistilBERT-base-uncased (Compact Baseline)   | - Qwen2.5-0.5B-Instruct (Compact SFT/ONNX)             |
| - DeBERTa-v3-base-FEVER (Fact Verification)    | - Llama-3.2-1B-Instruct (SOTA 1B Instruction LLM)      |
| - Llama-Guard-3-1B (2024 Safety Guardrail)     | - Phi-3.5-mini-instruct (3.8B Lab Frontier Ceiling)    |
+------------------------------------------------+--------------------------------------------------------+
| External Reference Baseline: Deterministic Rule Engine Oracle (Arm F of Factorial Ablation)             |
+---------------------------------------------------------------------------------------------------------+
```

---

## 2. Why We Do Not Use a Single Leaderboard

A frequent methodological error in applied deep learning research is ranking heterogeneous models on a single aggregated accuracy score. In our system:
- A sequence classification cross-encoder predicting whether an invoice is in an assessment's declared provenance ancestry solves a **binary edge classification problem**.
- A 3-class NLI model evaluating whether a policy clause entails an adjustor's claim solves a **semantic alignment problem**.
- A decoder-only language model generating structured JSON proposals solves an **instruction-following schema adherence problem**.
- A deterministic rule engine executes **exact symbolic predicate logic**.

Comparing these models on raw accuracy or a single blended score is scientifically invalid. Therefore, all comparisons are conducted strictly within task groups on identical evaluation splits.

---

## 3. Group 1: Semantic Reasoning and NLI

**Purpose:** Determine whether two statements entail, contradict, or are neutral to one another.

### Model 1: `cross-encoder/nli-MiniLM2-L6-H768`
- **Architecture:** 6-layer BERT-style cross-encoder ($H=768$, 12 heads), 66M parameters.
- **Publication:** Wang et al., "MiniLMv2" (Dec 2020 / ACL Findings 2021). Checkpoint: 2021.
- **Why Selected:** Historical repository baseline; measures minimum latency (~15ms) and memory footprint (<450MB RAM) for cross-attention.
- **Role:** Fast preliminary consistency screening between claim assertions and policy terms.

### Model 2: `cross-encoder/nli-deberta-v3-small`
- **Architecture:** Disentangled attention encoder + ELECTRA-style pretraining, 141M parameters.
- **Publication:** He et al., "DeBERTaV3" (Nov 2021 / ICLR 2023). Checkpoint: 2021/2022.
- **Why Selected:** Pinned primary baseline in repository (`semantic_advice.py`); provides state-of-the-art syntactic reasoning for small encoders.
- **Role:** Standard transformer-based NLI comparator against which custom models and ModernBERT are evaluated.

### Model 3: `answerdotai/ModernBERT-base`
- **Architecture:** Modernized bidirectional encoder with RoPE, GeGLU, FlashAttention-2, and unpadding, 149M parameters.
- **Publication:** Warner et al., "Smarter, Better, Faster, Longer" (Dec 2024). Checkpoint: Dec 2024.
- **Why Selected:** Modern 2024 architecture with native 8,192 token context window; eliminates the 512-token truncation limit of BERT/DeBERTa when reading multi-page insurance contracts.
- **Role:** Long-context policy verification engine reading full contracts without truncation.

**Group Datasets:** MultiNLI (general out-of-domain robustness), ContractNLI (contractual and policy clauses), and repository diagnostic smoke pairs (`baseline_tests.json`).

---

## 4. Group 2: Computational Dependency Prediction

**Purpose:** Predict whether a specific assessment depends on another artifact or piece of evidence.

### Model 4: `TF-IDF + Calibrated Logistic Regression`
- **Architecture:** N-gram TF-IDF vectorizer (unigrams + bigrams) + L2-regularized Logistic Regression with Platt scaling, ~50k sparse parameters.
- **Publication:** Pedregosa et al., "Scikit-learn" (JMLR 2011, updated 2023).
- **Why Selected:** Classical non-neural baseline; measures whether deep cross-attention adds real value over lexical word overlap.
- **Role:** Non-neural lower bound benchmark for dependency edge detection.

### Model 5: `CUSTOM DeBERTa-v3-small Dependency Model`
- **Architecture:** DeBERTa-v3-small encoder backbone + custom 2-layer MLP classification head, 141M parameters.
- **Publication:** He et al. (2021); project-specific adaptation and fine-tuning (2026). Checkpoint: `_private/runs/candidate_v2_*/checkpoints/best`.
- **Why Selected:** Core custom research model; predicts non-authoritative candidate dependency edges between updated sources and assessments.
- **Role:** Subject model evaluated across backbone initialization, 1-epoch adaptation, historical V2 checkpoint, and V3 root-explicit contrast variant.

### Model 6: `BAAI/bge-small-en-v1.5`
- **Architecture:** 12-layer MiniLM-style bi-encoder producing 384-dimensional dense vectors, 33.5M parameters.
- **Publication:** Xiao et al., "C-Pack" (Sep 2023 / EMNLP 2024). Checkpoint: 2023/2024.
- **Why Selected:** Dense bi-encoder comparator; tests whether $O(1)$ pre-computed vector cosine similarity can surrogate for cross-encoders.
- **Role:** Evaluates whether cached vector embeddings can accurately detect dependencies without cross-attention.

**Group Datasets:** Candidate V2 controlled dataset (568 rows) and Candidate V3 root-explicit contrast suite (512 rows).  
**Primary Metric:** **False Negative Rate ($\text{FNR}_{\text{dep}}$)**, because missing a dependency causes a stale certificate to be reused!

---

## 5. Group 3: Evidence Verification and Fault Detection

**Purpose:** Detect unsupported claims, contradictions, and unreliable agent assessments.

### Model 7: `distilbert/distilbert-base-uncased`
- **Architecture:** 6-layer knowledge-distilled transformer encoder, 66M parameters.
- **Publication:** Sanh et al., "DistilBERT" (Oct 2019). Checkpoint: 2019.
- **Why Selected:** Trainable compact transformer baseline replacing the deterministic rule oracle in the 12 trainable model slots.
- **Role:** Compact evidence classification baseline establishing minimum parameter requirements for fact checking.

### Model 8: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
- **Architecture:** DeBERTa-v3-base cross-encoder explicitly pre-trained on FEVER fact-verification, 435M parameters.
- **Publication:** Laurer et al. (EMNLP 2022 / 2024). Checkpoint: 2023/2024.
- **Why Selected:** Strongest encoder for fact checking; pre-trained on FEVER to detect factual discrepancies between text pairs.
- **Role:** Primary fact-checking specialist model intercepting ungrounded claims before consensus insertion.

### Model 9: `meta-llama/Llama-Guard-3-1B`
- **Architecture:** 16-layer transformer decoder with safety classification heads (Llama-3.2 backbone), 1.23B parameters.
- **Publication:** Llama Team, "Llama Guard 3" (Meta AI, Sep 2024). Checkpoint: Sep 2024.
- **Why Selected:** Specialized 2024 guardrail model; classifies whether agent proposals contain policy violations or corrupt reasoning.
- **Role:** Advisory output inspector detecting policy violations, malformed schema, and prompt injection attempts.

*External Reference Baseline:* **Deterministic Rule Engine Oracle** (Arm F of Factorial Ablation; 0 parameters; 100% decided safety ceiling).

---

## 6. Group 4: Generative Specialist Agents

**Purpose:** Generate structured assessments for policy, document, risk, and compliance specialists.

### Model 10: `Qwen/Qwen2.5-0.5B-Instruct`
- **Architecture:** 24-layer transformer decoder (RoPE, SwiGLU, RMSNorm, 32k context), 494M parameters.
- **Publication:** Qwen Team, "Qwen2.5" (Sep 2024). Checkpoint: Sep 2024 (INT8 ONNX in repo).
- **Why Selected:** Integrated in repository (`model_agent.py`); proven ability to generate valid JSON claim assessments on CPU.
- **Role:** Primary lightweight generative specialist agent drafting structured proposals under independent validator inspection.

### Model 11: `meta-llama/Llama-3.2-1B-Instruct`
- **Architecture:** 16-layer transformer decoder with Grouped-Query Attention (GQA) and 128k context, 1.23B parameters.
- **Publication:** Dubey et al., "The Llama 3 Herd of Models" (Meta AI, Sep 2024). Checkpoint: Sep 2024.
- **Why Selected:** SOTA open-weight 1B instruction model; tests if scaling to 1B eliminates 0.5B logical reasoning flaws and prompt sensitivity.
- **Role:** Intermediate generative specialist agent balancing reasoning depth against inference throughput.

### Model 12: `microsoft/Phi-3.5-mini-instruct`
- **Architecture:** 32-layer transformer decoder with synthetic textbook pretraining and DPO, 3.82B parameters.
- **Publication:** Abdin et al., "Phi-3 Technical Report" (Microsoft, Aug 2024). Checkpoint: Aug 2024.
- **Why Selected:** Laboratory frontier reasoning ceiling; tests the upper performance bound of compact open-weight models.
- **Role:** Theoretical performance ceiling in university laboratory experiments; strictly marked `DEFERRED_TO_NVIDIA`.

---

## 7. Summary Matrix

The complete 17-column dataset is preserved in [12_model_research_matrix.xlsx](../../12_model_research_matrix.xlsx) and [results/Model_Journey_For_Mentor.xlsx](../../results/Model_Journey_For_Mentor.xlsx).
