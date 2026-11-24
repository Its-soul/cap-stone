# Experimental Design: Fair Comparative Evaluation of 12 AI/ML Models
**Project:** Verifiable Multi-Agent Fault-Tolerant Adjudication System with Cryptographic Causal Provenance  
**Repository:** [https://github.com/Its-soul/cap-stone](https://github.com/Its-soul/cap-stone)  
**Investigation Date:** October 2026  
**Author:** Senior AI/ML Research Scientist, Deep Learning Specialist & MLOps Engineer  

---

## 1. Scientific Philosophy & Core Precepts

A fatal flaw in many multi-agent AI research papers is forcing heterogeneous model architectures into a single monolithic leaderboard. In this capstone project:
- A sequence classification cross-encoder predicting whether an invoice is dependent on a policy rider solves a **binary edge classification problem**.
- A 3-class NLI model evaluating whether an insurance policy clause entails an adjustor’s conclusion solves a **semantic alignment problem**.
- A decoder-only language model generating structured JSON assessment proposals solves an **instruction-following, generation-with-schema problem**.
- A deterministic rule engine executes **exact symbolic predicate logic**.

Comparing these models on raw accuracy or a single blended score is scientifically invalid. Consequently, this experimental design partitions our 12 shortlisted models into **four rigorously separated, task-aligned experimental groups**:

```
+---------------------------------------------------------------------------------------------------------+
|                                    FOUR COMPARATIVE EXPERIMENTAL GROUPS                                 |
+---------------------------------------------------------------------------------------------------------+
| Group 1: Semantic Reasoning & NLI              | Group 2: Computational Dependency Prediction           |
| (3-class Cross-Encoder Sequence Pair Classif.) | (Binary Edge Classification on Declared Provenance)    |
| - MiniLM2-L6-H768 (Compact Baseline)           | - TF-IDF + Logistic Regression (Linear Baseline)       |
| - DeBERTa-v3-small (Pinned Primary Baseline)   | - DeBERTa-v3-small-dep (Neural Sequence Classifier)    |
| - ModernBERT-base-nli (2024 SOTA 8k Context)   | - BGE-small-en-v1.5 (Dense Bi-Encoder Baseline)        |
+------------------------------------------------+--------------------------------------------------------+
| Group 3: Evidence & Fault Detection            | Group 4: Specialist Assessment Generation              |
| (Unsupported Claim & Guardrail Classification) | (Structured JSON Generation & Verifier Acceptance)     |
| - Rule Engine Oracle (Deterministic Baseline)  | - Qwen2.5-0.5B-Instruct (Compact SFT/ONNX)             |
| - DeBERTa-v3-base-FEVER (Fact Verification)    | - Llama-3.2-1B-Instruct (SOTA 1B Instruction LLM)      |
| - Llama-Guard-3-1B (2024 Safety Guardrail)     | - Phi-3.5-mini-instruct (3.8B Lab Frontier Ceiling)    |
+------------------------------------------------+--------------------------------------------------------+
```

---

## 2. Group 1: Natural Language Inference (NLI) & Semantic Reasoning

### 2.1. Research Question (RQ1)
> *Does upgrading from standard MiniLM/DeBERTa encoders to a modern long-context encoder (ModernBERT) improve semantic entailment and contradiction detection across complex, multi-page insurance policy clauses without suffering truncation degradation?*

### 2.2. Task Definition & Input-Output Contract
- **Input:** Premise $P$ (Policy text or contractual clause) and Hypothesis $H$ (Adjustor statement or claim assertation).
- **Output:** Categorical distribution over 3 classes: $\hat{y} \in \{\text{entailment}, \text{neutral}, \text{contradiction}\}$.
- **Loss Function:** Multi-class Cross-Entropy Loss:
  $$\mathcal{L}_{\text{CE}} = - \sum_{c=1}^3 y_c \log p_c$$

### 2.3. Models Evaluated
- **Baseline:** `cross-encoder/nli-MiniLM2-L6-H768` (66M params, Wang et al. 2020)
- **Primary Pinned Comparator:** `cross-encoder/nli-deberta-v3-small` (141M params, He et al. 2021)
- **Proposed SOTA Upgrade:** `tasksource/ModernBERT-base-nli` (149M params, Warner et al. Dec 2024, 8k context)

### 2.4. Dataset & Partitioning
- **Primary Benchmarks:** MultiNLI (General out-of-domain robustness) + ContractNLI (Domain-specific contractual clauses).
- **Repository Smoke Suite:** `baseline_tests.json` (6 author-annotated pairs, including stress pair `nli-003`).
- **Splits:** 80% train, 10% validation, 10% held-out test (partitioned by document hash).

### 2.5. Evaluation Metrics
1. **Macro-Averaged F1 Score ($\text{F1}_{\text{macro}}$):** Balances performance across entailment, neutral, and contradiction.
2. **Contradiction Class Precision ($P_{\text{contra}}$):** Crucial for fraud and exclusion detection; false contradictions cause unwarranted claim rejections.
3. **Truncation Penalty Curve:** Model accuracy measured as input sequence length scales from 256 to 4,096 tokens (exposing DeBERTa’s 512-token cutoff vs ModernBERT’s native 8,192 capacity).
4. **Expected Calibration Error (ECE):** Assesses whether predicted probabilities reflect true empirical error rates.

### 2.6. Expected Failure Modes & Statistical Tests
- **Failure Mode:** Lexical mismatch penalty (treating dissimilar neutral concepts as contradictory, as observed in `baseline_tests.json:nli-003`).
- **Statistical Significance:** McNemar's test on paired classification predictions ($\alpha = 0.01$) and 95% bootstrap confidence intervals (1,000 resamples).

---

## 3. Group 2: Computational & Causal Dependency Prediction

### 3.1. Research Question (RQ2)
> *Can neural cross-encoders trained on root-explicit contrastive provenance data learn true computational dependency edges between upstream artifacts and downstream assessments, or do they merely exploit surface semantic similarity? How do they compare against a classical TF-IDF baseline and a dense bi-encoder?*

### 3.2. Task Definition & Input-Output Contract
- **Input:** Upstream source artifact content $A_{\text{src}}$ (with explicit root ancestry closure) and downstream candidate assessment $A_{\text{cand}}$.
- **Output:** Binary prediction $\hat{y} \in \{0 (\text{independent}), 1 (\text{depends\_on})\}$.
- **Decision Rule:** Output edge predicted if $P(\text{depends\_on} \mid A_{\text{src}}, A_{\text{cand}}) \ge \tau$, where $\tau \in [0.1, 0.5]$ is calibrated on validation data.

### 3.3. Models Evaluated
- **Classical ML Baseline:** `TF-IDF + Calibrated Logistic Regression` (Pedregosa et al. 2011)
- **Dense Bi-Encoder Baseline:** `BAAI/bge-small-en-v1.5` (33.5M params, Xiao et al. 2023)
- **Deep Cross-Encoder:** `DeBERTa-v3-small-dep` (141M params, Candidate V2 vs Candidate V3 trained weights)

### 3.4. Dataset & Split Strategy
- **Primary Corpus:** Extended Repository Provenance Suite (`controlled_data.py` V2 suite: 568 rows; `v3_preparation.py` root-explicit contrast suite: 512 rows).
- **Total Combined:** 1,080 annotated provenance pairs.
- **Partitioning:** Hash-partitioned on `claim_id` into Train (512), Validation (128), Stress Contrast (248), and Held-Out Final (192).

### 3.5. Evaluation Metrics & The Missed Dependency Asymmetry
1. **Missed Dependency False Negative Rate ($\text{FNR}_{\text{dep}}$) - CRITICAL METRIC:**
   $$\text{FNR}_{\text{dep}} = \frac{\text{False Negatives}}{\text{True Positives} + \text{False Negatives}}$$
   *Rationale:* In selective recovery, a **False Positive** (predicting dependency where none exists) merely triggers an unnecessary recomputation (costing CPU cycles). However, a **False Negative** (missing a real dependency) causes a stale, corrupted certificate to be silently reused, **violating cryptographic audit integrity**. Therefore, $\text{FNR}_{\text{dep}}$ must be strictly minimized.
2. **Macro-F1 & Precision-Recall AUC (PR-AUC):** Accounts for class balance between independent and dependent candidate pairs.
3. **NLI Confound Error Rate:** Specifically measured on the 16 `nli_context` contrast pairs where semantic facts overlap but no DAG edge exists.

### 3.6. Expected Failure Modes & Statistical Tests
- **Failure Mode:** Semantic overlap shortcutting (predicting dependency whenever an assessment quotes an invoice, ignoring whether the assessment logic was formally derived from it).
- **Statistical Significance:** Paired permutation test (10,000 permutations) on $\text{FNR}_{\text{dep}}$ and Macro-F1.

---

## 4. Group 3: Evidence & Fault Detection

### 4.1. Research Question (RQ3)
> *Can dedicated fact-verification (FEVER) and guardrail models reliably intercept hallucinated specialist rationales, malformed schema, and forged artifacts before they enter the adjudication consensus layer? How close do they come to the 100% safety ceiling of deterministic rule gates?*

### 4.2. Task Definition & Input-Output Contract
- **Input:** Attached evidentiary documents $E$ (medical invoices, police reports, proof of loss) and specialist agent output proposal $O = \{ \text{decision}, \text{rationale}, \text{cited\_amounts} \}$.
- **Output:** Verification label $\hat{y} \in \{\text{VERIFIED}, \text{UNSUPPORTED / CORRUPTED}\}$.

### 4.3. Models Evaluated
- **Deterministic Oracle Baseline:** `Deterministic Rule Engine Oracle` (Arm F of Factorial Ablation; 0 params)
- **Fact-Verification Cross-Encoder:** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (435M params, Laurer et al. 2022)
- **Generative Safety Guardrail:** `meta-llama/Llama-Guard-3-1B` (1.23B params, Meta AI Sep 2024)

### 4.4. Dataset & Split Strategy
- **Corpus:** FEVER Fact Verification benchmark + Synthetic Corrupted Agent Outputs (`controlled_data.py` corrupted artifact generator injecting numeric hallucinations, mismatched policy IDs, forged timestamps, and malformed JSON).
- **Total Suite:** 600 paired verification instances (300 benign, 300 perturbed/adversarial).

### 4.5. Evaluation Metrics
1. **Unsupported Claim Interception Rate (Recall on Faults):** Percentage of corrupted or hallucinated outputs successfully flagged and prevented from receiving certificates.
2. **False Veto Rate (Fall-out):** Percentage of perfectly valid specialist outputs falsely flagged, forcing unnecessary human `REVIEW`.
3. **End-to-End Consensus Safety:** Measured in simulated adjudication panels: count of Unsafe Approvals (Target = 0).

---

## 5. Group 4: Specialist Assessment Generation

### 5.1. Research Question (RQ4)
> *Can small open-weight generative language models (0.5B to 3.8B parameters) reliably generate syntactically valid JSON proposals with high evidence fidelity and policy correctness under prompt constraints, and does scaling from 0.5B to 3.8B significantly improve verifier acceptance rates in multi-agent adjudication?*

### 5.2. Task Definition & Input-Output Contract
- **Input:** Unstructured insurance claim dossier (incident description, claimed damages, scanned bill text, policy clauses) formatted into a structured prompt with 2-shot domain demonstrations.
- **Output:** Valid JSON object adhering to strict domain schema:
  ```json
  {
    "domain": "coverage-v1",
    "decision": "APPROVE" | "REJECT" | "REVIEW",
    "confidence": 0.85,
    "rationale": "Claimant was covered under Policy POL-2024-001 on the date of loss.",
    "evidence_citations": ["POL-2024-001:Section 4B"],
    "red_flags": []
  }
  ```

### 5.3. Models Evaluated
- **Compact Integrated Baseline:** `Qwen/Qwen2.5-0.5B-Instruct` (INT8 ONNX / SFT, 494M params, Alibaba 2024)
- **SOTA 1B Instruction Comparator:** `meta-llama/Llama-3.2-1B-Instruct` (1.23B params, Meta AI Sep 2024)
- **Laboratory Frontier Ceiling:** `microsoft/Phi-3.5-mini-instruct` (3.82B params, Microsoft Aug 2024)

### 5.4. Dataset & Split Strategy
- **Corpus:** 100 Comprehensive Insurance Adjudication Scenarios (covering property, auto, and medical claims with complex exclusion riders).
- **Splits:** 50 scenarios for 2-shot few-shot prompt calibration and supervised fine-tuning (SFT); 50 held-out scenarios for blind multi-agent evaluation.

### 5.5. Evaluation Metrics
1. **JSON Syntax Validity Rate ($\%_{\text{JSON}}$):** Percentage of generations that parse cleanly into `pydantic` schema without repair (Target = 100%).
2. **Evidence Fidelity Score ($\%_{\text{Fidelity}}$):** Verification that every dollar amount and date cited in the rationale matches the source document verbatim (zero numeric hallucination).
3. **Verifier Acceptance Rate ($\%_{\text{Accept}}$):** Percentage of generated proposals accepted by `DeterministicValidator` without triggering rejection or fallback to human `REVIEW`.
4. **Single-Agent Inference Latency ($T_{\text{gen}}$):** Mean wall-clock time per proposal (seconds).

---

## 6. Addressing the 10 Research Validity Questions (Phase 8 Synthesis)

1. **Does improving NLI accuracy improve actual dependency detection?**  
   *No, not automatically.* NLI measures semantic entailment, not causal lineage. In our Candidate V2 audit, an NLI model with 91.7% accuracy consistently failed on 16 `nli_context` test pairs because it confused literal semantic truth with declared provenance support rules.
2. **Can models distinguish evidence relationships from computational dependencies?**  
   *Only when conditioned on explicit ancestral closure.* Models trained solely on standard text confuse "evidence cited" with "causal parentage". Group 2 evaluates whether explicit contrastive training (`Candidate V3`) eliminates this shortcut.
3. **Are learned dependency predictions safe enough to guide non-authoritative recommendations?**  
   *Yes, but only in an advisory capacity.* A learned dependency prediction can prioritize which nodes to inspect during an audit or suggest candidate graph edges, but it must **never** create an authoritative DAG edge or supersede deterministic graph traversal.
4. **Does evidence verification reduce unsupported specialist outputs?**  
   *Yes.* Incorporating FEVER-trained cross-encoders (Group 3) provides an independent layer that intercepts hallucinated citations before proposals reach consensus voting.
5. **Do newer models consistently outperform smaller baselines?**  
   *Modern architectures (ModernBERT, Llama-3.2) provide substantial structural benefits (8k context, GQA), but smaller models (MiniLM, TF-IDF) remain unmatched in inference latency and memory efficiency for constrained tasks.*
6. **How much computational cost is added by stronger models?**  
   *Empirically measured:* Moving from MiniLM (~15ms) to DeBERTa-base (~180ms) adds a 12x latency penalty. Moving to Qwen2.5-0.5B on CPU adds ~37,000ms (2,400x penalty). This quantified cost justifies using selective recovery to eliminate redundant LLM calls.
7. **Does the final adjudication system benefit from model improvements when mandatory rules remain authoritative?**  
   *Yes.* While mandatory rules prevent catastrophic unsafe approvals (as proven in the 150-run ablation study), high-quality model proposals reduce false reviews and improve nuanced assessment quality for borderline claims that pass mandatory checks.
8. **Can selective recovery preserve valid work after changes?**  
   *Yes.* In single-agent failure scenarios, selective recovery preserves 75% of valid specialist certificates ($v_1$), saving ~110 seconds of LLM execution time per claim during local adjudication.
9. **How do model failures influence REVIEW, APPROVE, and REJECT outcomes?**  
   *All unhandled model errors (invalid JSON, timeouts, confidence below threshold) are routed deterministically into `CONFIRMED_FAILURE`, triggering a safe fallback to `REVIEW`. No model failure can cause an erroneous `APPROVE`.*
10. **Which improvements are supported by experiments, and which remain hypotheses?**  
    - *Empirically Supported:* Rule Gate +48.2% accuracy jump, elimination of 109 unsafe approvals, 100% valid JSON with 2-shot Qwen2.5-0.5B, selective recovery state preservation.
    - *Current Hypotheses:* Whether Candidate V3 eliminates the NLI dependency confound; whether ModernBERT's 8k window outperforms chunked DeBERTa on full insurance policies.

---

## 7. Threats to Validity & Mitigation Strategies

| Threat to Validity | Nature of Risk | Concrete Mitigation Protocol |
| :--- | :--- | :--- |
| **Data Leakage Across Splits** | Model memorizes specific claim IDs or policy clauses from training split. | Strict hash-partitioning (`hash(claim_id) % 100`) ensures complete document-level isolation between train, validation, and test sets. |
| **Synthetic Artifact Homogeneity** | Models learn idiosyncratic patterns of synthetic data generators. | Incorporating real-world public benchmarks (MultiNLI, ContractNLI, FEVER, CUAD) alongside synthetic repository suites. |
| **Threshold Overfitting** | Tuning dependency classification threshold $\tau$ on the final test set. | Strict separation: threshold $\tau$ is tuned exclusively on validation splits using grid search; final test set is evaluated exactly once with frozen weights and thresholds. |
| **Stochastic Generation Variance** | Generative LLMs produce different decisions across runs. | Fix random seeds (`42`, `101`, `2024`), set temperature $T = 0.0$ (greedy decoding) for evaluation, and report standard deviation across runs. |
