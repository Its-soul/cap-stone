# Mentor Presentation Summary: Model Auditing and Selection Strategy
**Project:** Verifiable Multi-Agent Fault-Tolerant Adjudication System with Cryptographic Causal Provenance  
**Academic Degree:** Final-Year B.Tech in Computer Science & Engineering (AI/ML)  
**Author:** Student Research Team (Senior AI/ML Research Scientist & Deep Learning Lead)  
**Supervising Faculty:** University Project Mentor / Head of Department  
**Investigation Date:** October 2026  

---

## 1. Executive Briefing for Faculty Mentor

Respected Mentor,

This document summarizes our systematic investigation of the repository and presents our proposed research roadmap. In accordance with your guidance, we are **not immediately launching heavy model training**. Instead, we have completed a comprehensive audit of existing code, datasets, and historical experiments, and we have formulated a scientifically sound, resource-conscious strategy.

### The Core Research Problem
Modern automated insurance claim adjudication systems face severe challenges with reliability:
1. Large Language Models (LLMs) hallucinate facts and invent non-existent policy coverage.
2. When an invoice or police report changes, existing systems either re-run all AI agents from scratch (wasting immense compute) or blindly trust stale cached results (causing incorrect claim payouts).

### Our Architectural Solution
Our project implements a **hybrid adjudication architecture**:
- **Authoritative Deterministic Layer:** A Directed Acyclic Graph (DAG) state machine that manages certificates, enforces non-negotiable policy exclusion rules (e.g., expired policy = immediate rejection), chains events with SHA-256 hashes, and deterministically invalidates only the specific downstream calculations affected by an updated document.
- **Untrusted Advisory AI/ML Layer:** Fast, specialized AI models that suggest whether two documents are contradictory, predict which assessments depend on an invoice, and draft initial claim reviews.

> **Crucial Rule:** AI predictions carry **zero authoritative weight**. An AI model cannot issue a cryptographic certificate or override a mandatory policy rule. If an AI model fails or produces malformed output, the system safely falls back to human `REVIEW` without making an unauthorized payout.

---

## 2. Why We Need 12 Models: Four Task Groups

To ensure a fair and scientifically defensible comparison, we have not lumped all models into a single arbitrary leaderboard. Instead, we have shortlisted **12 models divided into 4 distinct task groups**, comparing each model only against valid alternatives solving the exact same problem:

```
+---------------------------------------------------------------------------------------------------+
| GROUP 1: Semantic Reasoning & Natural Language Inference (NLI)                                    |
| Goal: Detect whether an adjustor's claim contradicts an insurance policy clause.                  |
| Models: 1. MiniLM-L6 | 2. DeBERTa-v3-small | 3. ModernBERT-base                                   |
+---------------------------------------------------------------------------------------------------+
| GROUP 2: Computational & Causal Dependency Prediction                                             |
| Goal: Predict whether a claim assessment actually depends on a specific source invoice.            |
| Models: 4. TF-IDF + Logistic Reg | 5. DeBERTa-v3-small-dep | 6. BGE-small-en-v1.5 Bi-Encoder        |
+---------------------------------------------------------------------------------------------------+
| GROUP 3: Evidence Verification & Fault Detection                                                  |
| Goal: Intercept hallucinated citations or corrupted JSON before they reach consensus.             |
| Models: 7. Deterministic Rule Engine Oracle | 8. DeBERTa-FEVER | 9. Llama-Guard-3-1B              |
+---------------------------------------------------------------------------------------------------+
| GROUP 4: Specialist Assessment Proposal Generation                                                |
| Goal: Read unstructured claim notes and generate clean, valid structured JSON proposals.          |
| Models: 10. Qwen2.5-0.5B-Instruct | 11. Llama-3.2-1B-Instruct | 12. Phi-3.5-mini-instruct           |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. Plain-Language Explanation of Each Shortlisted Model

Below are concise answers to the 6 core questions for each of our 12 candidates, explained using our insurance claim scenario:

---

### Model 1: `cross-encoder/nli-MiniLM2-L6-H768`
1. **What is it?** A tiny, ultra-fast 6-layer BERT-style neural network (66 million parameters).
2. **When was it published?** Published in late 2020 by Microsoft Research; checkpoint released in 2021.
3. **Why are we using it?** It is our historical baseline in the repository. It gives us a benchmark for the minimum latency and memory footprint required for sentence comparison.
4. **What dataset will train it?** Pretrained on MultiNLI and evaluated on our domain smoke tests (`baseline_tests.json`).
5. **How will it be evaluated?** Macro-F1 score and inference latency (milliseconds per sentence pair).
6. **Why might it be better than alternatives?** It runs in just 15 milliseconds on a normal laptop CPU and uses less than 250 MB of memory.

---

### Model 2: `cross-encoder/nli-deberta-v3-small`
1. **What is it?** A specialized transformer (141 million parameters) that separates word meanings from their relative positions.
2. **When was it published?** Published in late 2021 by Microsoft Research (ICLR 2023).
3. **Why are we using it?** It is the pinned primary baseline in our repository. It provides significantly higher semantic accuracy than MiniLM.
4. **What dataset will train it?** MultiNLI and ContractNLI (legal and policy clauses).
5. **How will it be evaluated?** Macro-F1 and contradiction detection precision.
6. **Why might it be better than alternatives?** Its disentangled attention architecture catches subtle phrasing differences in insurance exclusions that simpler models miss.

---

### Model 3: `tasksource/ModernBERT-base-nli`
1. **What is it?** A state-of-the-art bidirectional transformer released in December 2024 (149 million parameters) with a native 8,192-token context window.
2. **When was it published?** Published in December 2024 by Answer.AI and LightOn.
3. **Why are we using it?** Traditional BERT and DeBERTa models cut off at 512 words. ModernBERT allows our system to read an entire 10-page insurance contract at once without truncating text.
4. **What dataset will train it?** ContractNLI and long-context insurance policy documents.
5. **How will it be evaluated?** Accuracy on long documents and resistance to truncation errors.
6. **Why might it be better than alternatives?** It processes 8,000 words natively and runs up to 2.5x faster per token than older encoders.

---

### Model 4: `TF-IDF + Calibrated Logistic Regression`
1. **What is it?** A classical, transparent machine learning algorithm that counts word frequencies and fits a calibrated linear curve.
2. **When was it published?** Scikit-learn foundational release (2011, updated 2023).
3. **Why are we using it?** It is our non-neural baseline. In scientific research, we must prove that a deep neural network actually beats simple word counting.
4. **What dataset will train it?** Our repository's Candidate V2 and V3 provenance dependency suites.
5. **How will it be evaluated?** False Negative Rate on missed dependencies, Macro-F1, and training speed.
6. **Why might it be better than alternatives?** It trains in less than 2 seconds, requires zero GPU memory, and is 100% explainable (we can see exactly which words triggered a dependency flag).

---

### Model 5: `DeBERTa-v3-small-dep` (Trained Sequence Classifier)
1. **What is it?** A DeBERTa-v3 model with a custom classification head fine-tuned to predict whether an assessment depends on a source artifact.
2. **When was it published?** Backbone from 2021; custom dependency head trained in our repository in 2026.
3. **Why are we using it?** It is the model we previously trained (Candidate V2). We need to test it on our new Candidate V3 contrast dataset to see if we cured its past confusion between semantic similarity and true dependency.
4. **What dataset will train it?** Candidate V3 Root-Explicit Contrast Suite (512 rows with strict ancestral closure rules).
5. **How will it be evaluated?** False Negative Rate (missed dependencies must be near zero) and Macro-F1.
6. **Why might it be better than alternatives?** It cross-attends directly between the entire invoice content and the assessment text simultaneously.

---

### Model 6: `BAAI/bge-small-en-v1.5` (Dense Bi-Encoder)
1. **What is it?** A compact embedding model (33.5 million parameters) that converts text into 384-dimensional mathematical vectors.
2. **When was it published?** Published in late 2023 by the Beijing Academy of Artificial Intelligence (EMNLP 2024).
3. **Why are we using it?** To test whether we can predict dependencies simply by checking if two document vectors point in the same direction (cosine similarity).
4. **What dataset will train it?** Candidate V2 and V3 provenance datasets.
5. **How will it be evaluated?** ROC-AUC and Macro-F1 across varying similarity thresholds.
6. **Why might it be better than alternatives?** Document embeddings can be calculated once and stored. Comparing them takes microsecond lookups ($O(1)$), which would make selective recovery instant.

---

### Model 7: `Deterministic Rule Engine Oracle`
1. **What is it?** Pure symbolic logic and hard-coded Boolean policy rules (0 parameters; no neural weights).
2. **When was it published?** Implemented in our repository in 2026 (`src/cert_recovery/consensus.py`).
3. **Why are we using it?** It represents Arm F of our ablation study and establishes our ground-truth safety ceiling.
4. **What dataset will train it?** None (it uses exact policy logic: `if policy_expired: return REJECT`).
5. **How will it be evaluated?** Zero Unsafe Approvals across 150 stress scenarios.
6. **Why might it be better than alternatives?** It has 0% chance of hallucinating, runs in under 1 millisecond, and is legally infallible on explicit rules.

---

### Model 8: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
1. **What is it?** A DeBERTa model (435 million parameters) explicitly fine-tuned on the FEVER fact-checking dataset.
2. **When was it published?** Published in 2022 by Laurer et al. (updated 2024).
3. **Why are we using it?** To check whether the factual assertions made by an AI specialist are actually supported by the attached medical invoice or police report.
4. **What dataset will train it?** FEVER fact verification corpus and annotated insurance claim-evidence pairs.
5. **How will it be evaluated?** Unsupported claim interception rate (detecting when an AI quotes a dollar amount not found in the bill).
6. **Why might it be better than alternatives?** It was trained specifically to detect factual discrepancies rather than general grammatical relationships.

---

### Model 9: `meta-llama/Llama-Guard-3-1B`
1. **What is it?** A 1.2-billion-parameter safeguard model designed by Meta in late 2024 to classify safety and policy compliance.
2. **When was it published?** Published in September 2024 by Meta AI.
3. **Why are we using it?** To act as an automated inspector that scans an agent's output for policy violations or abnormal reasoning before it enters the consensus DAG.
4. **What dataset will train it?** HaluEval hallucination corpus and synthetic corrupted claim outputs.
5. **How will it be evaluated?** Precision and recall on detecting corrupted, invalid, or policy-violating recommendations.
6. **Why might it be better than alternatives?** It understands nuanced human language violations and complex instruction constraints far better than simple regex filters.

---

### Model 10: `Qwen/Qwen2.5-0.5B-Instruct`
1. **What is it?** A compact, state-of-the-art generative language model (494 million parameters).
2. **When was it published?** Released in September 2024 by Alibaba Cloud.
3. **Why are we using it?** It is already integrated into our project (`src/cert_recovery/model_agent.py`). It reads claim notes and drafts structured JSON recommendations.
4. **What dataset will train it?** InsuranceQA and 50 annotated claim adjudication scenarios (2-shot prompting / LoRA).
5. **How will it be evaluated?** Valid JSON syntax rate (100% required), evidence accuracy, and verifier acceptance rate.
6. **Why might it be better than alternatives?** It is under 500 MB in size and achieved 100% valid JSON adherence in our preliminary 2-shot tests on CPU.

---

### Model 11: `meta-llama/Llama-3.2-1B-Instruct`
1. **What is it?** Meta's flagship lightweight instruction-tuned language model (1.23 billion parameters) released in late 2024.
2. **When was it published?** Released in September 2024 by Meta AI.
3. **Why are we using it?** To test whether moving from 0.5B to 1B parameters eliminates the logical reasoning flaws and prompt sensitivity we observed in Qwen.
4. **What dataset will train it?** Insurance claim adjudication instructions and CUAD contract data.
5. **How will it be evaluated?** Multi-clause decision accuracy and verifier acceptance rate.
6. **Why might it be better than alternatives?** Grouped-Query Attention (GQA) allows fast generation, and its superior pretraining gives it much stronger deductive logic on complex claims.

---

### Model 12: `microsoft/Phi-3.5-mini-instruct`
1. **What is it?** A dense, high-capability reasoning model (3.82 billion parameters) trained by Microsoft on high-quality synthetic textbook data.
2. **When was it published?** Released in August 2024 by Microsoft Research.
3. **Why are we using it?** It serves as our "Laboratory Frontier Benchmark". It tells us how well the system would perform if we had unlimited server compute.
4. **What dataset will train it?** Multi-specialist claim adjudication corpus in the university lab.
5. **How will it be evaluated?** Deep reasoning accuracy and multi-agent consensus alignment.
6. **Why might it be better than alternatives?** It rivals much larger 7B–14B models on logic and factual reasoning, establishing the theoretical upper performance bound.

---

## 4. Two-Phase Execution Plan

### Phase 1: Personal Laptop Feasibility (CPU Only)
- **Goal:** Verify that models can load, train for 1 epoch on CPU, calculate gradients, and save checkpoints safely without crashing the laptop or using more than 3.5 GB of RAM.
- **Feasible on Laptop:** Models 1, 2, 3 (seq 512), 4, 5, 6, 7, 8, and 10 (LoRA).
- **Inference Smoke Only on Laptop:** Model 9 and Model 11 (tested on 10 cases to verify tokenizer and JSON outputs).
- **NVIDIA Lab Only:** Model 12 (3.82B parameters strictly reserved for the university GPU cluster).

### Phase 2: University NVIDIA GPU Laboratory (Extensive Benchmarking)
- **Goal:** Multi-epoch training (5–10 epochs), hyperparameter optimization using Optuna, triplicate random seeds (`42`, `101`, `2024`), mixed precision (`bf16`), and final evaluation on our sealed test datasets.
- **Expected Outcome:** High-impact empirical findings ready for submission to an AI/ML peer-reviewed conference or journal.
