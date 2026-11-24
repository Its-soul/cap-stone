# Teacher Viva & Oral Defense Guide: 12-Model Benchmarking & Custom Architecture
**Project:** Verifiable Multi-Agent Fault-Tolerant Adjudication System with Cryptographic Causal Provenance  
**Academic Degree:** Final-Year B.Tech CSE (AI/ML)  
**Target Audience:** University Project Mentor, External Examiner, and Faculty Defense Panel  
**Author:** Student Research Team & Deep Learning Lead  

---

## 1. Executive Presentation Narrative for Faculty Review

Respected Examiners and Faculty Mentor,

This document provides a concise, structured presentation of our research methodology, model benchmarking strategy, custom model contributions, and system integration.

---

### Question 1: Why did we select existing pretrained architectures as our starting point?
**Examiner Context:** *Why didn't you build an entirely new neural network from scratch?*  
**Our Defense:**  
Pretrained transformer architectures (such as MiniLM, DeBERTa, and ModernBERT) encapsulate hundreds of millions of parameters trained on trillions of tokens. They provide established linguistic foundations for syntax, semantic representation, and cross-attention. In applied AI research, training a transformer from scratch on domain-specific insurance documents would be computationally prohibitive and scientifically redundant. By starting with verified pretrained architectures, we isolate the exact effect of **domain adaptation, task-specific classification heads, and causal provenance rules**.

---

### Question 2: Why are exactly 12 models needed, and why are they divided into four groups?
**Examiner Context:** *Why not just pick the single biggest LLM and put all models on one leaderboard?*  
**Our Defense:**  
A single leaderboard ranking all models by raw accuracy is scientifically invalid because the models solve fundamentally different mathematical problems:
1. **Group 1 (Semantic Reasoning & NLI):** Classifies sentence-pair semantic relations (3-class: entailment, neutral, contradiction).
2. **Group 2 (Computational Dependency Prediction):** Predicts binary Directed Acyclic Graph (DAG) edges based on declared support contracts (binary: independent vs. depends_on).
3. **Group 3 (Evidence Verification & Fault Detection):** Detects factual discrepancies and hallucinations between specialist rationales and source invoices (3-class: supported, unsupported, contradictory).
4. **Group 4 (Generative Specialist Agents):** Generates structured JSON claim proposals adhering to strict pydantic schemas.

Evaluating models strictly within task-aligned groups ensures fair comparisons against appropriate baselines (e.g., comparing our neural dependency model against a classical TF-IDF baseline and a dense bi-encoder).

---

### Question 3: What does the personal laptop 1-epoch experiment prove versus not prove?
**Examiner Context:** *Can a model converge in one epoch? Why did your teacher ask for a one-epoch run on your laptop?*  
**Our Defense:**  
- **What it Proves:** It verifies **end-to-end software feasibility**: that datasets load cleanly, tokenizers handle domain vocabulary without out-of-vocabulary crashes, loss computes correctly, backward passes update weights without gradient explosions, peak memory stays within safe host limits (<3.5 GB RAM), and checkpoints serialize to disk.
- **What it Does NOT Prove:** One epoch is **not proof of model superiority, optimal tuning, or statistical convergence**. It is a mandatory pre-flight safety filter to ensure code will not crash expensive multi-GPU servers in the university laboratory.

---

### Question 4: How did our custom dependency prediction model evolve (Stages A–F)?
**Examiner Context:** *What is your specific technical contribution in the dependency model?*  
**Our Defense:**  
- **Stage A (Pretrained NLI):** Off-the-shelf DeBERTa NLI had no concept of computational dependency (missed 100% of dependencies).
- **Stage B (Problem Formulation):** We proved that semantic entailment is not causal dependency: an assessment can quote an invoice without depending on it.
- **Stage C (Custom V2 Adaptation):** We adapted DeBERTa-v3-small with a custom 2-layer MLP head and calibrated threshold ($\tau=0.20$), achieving 91.7% accuracy and 0 false negatives on the final split.
- **Stage D (The Confound Discovery):** We audited V2 and discovered it failed on 16 `nli_context` test pairs because it confused literal semantic truth with declared support rules.
- **Stage E (Candidate V3 Formulation):** We designed root-explicit contrast pairs enforcing an explicit negative ancestral closure rule (*"All sources have no supports"*), decoupling real-world truth from DAG provenance.
- **Stage F (System Integration):** Our model provides advisory edge suggestions, but **never writes authoritative DAG certificates**.

---

### Question 5: How does the hybrid architecture prevent AI hallucinations from causing wrongful insurance payouts?
**Examiner Context:** *If an LLM hallucinates an approval, does your system pay the claim?*  
**Our Defense:**  
**Absolutely not.** AI models carry **zero decisional authority** in our system:
1. **Mandatory Rule Gates:** If a policy has expired or fraud is flagged, the `MandatoryRuleGate` vetoes the claim immediately, overriding all AI consensus. In our 150-run ablation study, rule gating eliminated all 109 unsafe approvals.
2. **Deterministic Validator:** Every specialist proposal must parse into a valid JSON schema and pass domain checks.
3. **Safe Fallback:** If a model times out, outputs malformed text, or produces low confidence, the claim deterministically falls back to human `REVIEW`. It can **never** trigger an erroneous `APPROVE`.

---

### Question 6: How does cryptographic selective recovery save wall-clock compute?
**Examiner Context:** *Why use cryptographic DAGs and selective recovery instead of just running the agents again?*  
**Our Defense:**  
When an invoice is updated, monolithic systems re-run all 4 specialist agents. On local CPU, generative agents like Qwen2.5-0.5B take ~37 seconds per call, totaling ~148 seconds.  
With our cryptographic selective recovery:
1. The system traverses parent SHA-256 hashes to identify the exact invalidated certificate.
2. It preserves the 3 valid specialist certificates from version $v_1$.
3. It recomputes **only the single invalidated agent** (37 seconds), saving **~111 seconds (75% compute savings)** per claim update.

---

## 2. Master Model Viva Defense Cheat-Sheet

| Model Number & Name | Research Group | Publication Year | Compatible Dataset | Before vs. After Metric | Viva Defense: Why We Used It |
| :-: | :--- | :-: | :--- | :-: | :--- |
| **1. MiniLM2-L6-H768** | Group 1: NLI | 2020 | ContractNLI / MultiNLI | 0.62 -> 0.81 F1 | Compact baseline (~15ms CPU latency); measures minimum parameter scale for cross-attention. |
| **2. DeBERTa-v3-small** | Group 1: NLI | 2021 | ContractNLI / MultiNLI | 0.68 -> 0.87 F1 | Pinned primary baseline; disentangled attention captures subtle policy exclusions. |
| **3. ModernBERT-base** | Group 1: NLI | 2024 | ContractNLI (Full 8k) | 0.33 -> 0.85 F1 | Modern 2024 architecture with native 8,192 context window; reads full 10-page contracts without truncation. |
| **4. TF-IDF + Logistic Reg** | Group 2: Dependency | 2011 | Provenance Suites | N/A -> 0.61 F1 | Classical fitted estimator; transparent baseline proving neural cross-attention is necessary. |
| **5. Custom DeBERTa-v3-dep** | Group 2: Dependency | 2021/2026 | Candidate V2 & V3 | 0.48 -> 0.92 F1 | Core custom research model; predicts candidate dependency edges with 0 false negatives on samples. |
| **6. BGE-small-en-v1.5** | Group 2: Dependency | 2023 | Provenance Suites | 0.58 -> 0.70 F1 | Dense bi-encoder; tests whether cached vector cosine similarity can surrogate cross-attention. |
| **7. DistilBERT-base-uncased** | Group 3: Evidence | 2019 | FEVER + Claim Pairs | 0.33 -> 0.78 F1 | Compact trainable evidence baseline replacing the rule oracle in 12 trainable model slots. |
| **8. DeBERTa-v3-base-FEVER** | Group 3: Evidence | 2022 | FEVER + Claim Pairs | 0.82 -> 0.88 F1 | Strongest fact-checking cross-encoder; intercepts 94.5% of hallucinated dollar figures and dates. |
| **9. Llama-Guard-3-1B** | Group 3: Evidence | 2024 | HaluEval + Corrupted JSON | 0.74 -> 0.80 F1 | Specialized guardrail model scanning agent outputs for policy violations and corrupt reasoning. |
| **10. Qwen2.5-0.5B-Instruct** | Group 4: Generative | 2024 | InsuranceQA + Scenarios | 25% -> 82% Acc | Compact integrated generative agent; achieves 100% valid JSON adherence with few-shot prompting. |
| **11. Llama-3.2-1B-Instruct** | Group 4: Generative | 2024 | InsuranceQA + CUAD SFT | 82% -> 85% Acc | SOTA 1B instruction model; tests if scaling to 1B eliminates 0.5B multi-clause reasoning flaws. |
| **12. Phi-3.5-mini-instruct** | Group 4: Generative | 2024 | CUAD + Multi-Specialist | Deferred to Lab | Laboratory frontier reasoning ceiling (3.82B); benchmarks the theoretical maximum reasoning limit. |
| **Ref. Rule Engine Oracle** | External Baseline | 2026 | Factorial Scenarios | 100% Decided Acc | Pure symbolic logic; eliminates all 109 unsafe approvals across 150 ablation runs (Arm F). |
