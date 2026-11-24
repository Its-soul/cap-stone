# Dataset and Model Compatibility Report
**Project:** Verifiable Multi-Agent Fault-Tolerant Adjudication System with Cryptographic Causal Provenance  
**Repository:** [https://github.com/Its-soul/cap-stone](https://github.com/Its-soul/cap-stone)  
**Investigation Date:** October 2026  
**Auditor:** Senior AI/ML Research Scientist & MLOps Specialist  

---

## 1. Overview and Classification Taxonomy

A rigorous empirical study requires mapping datasets to clearly delineated research questions and model families. This report audits all **existing datasets** currently residing in the repository and specifies **curated external benchmark datasets** to support the 12-model comparative experimental plan.

Datasets are audited across six functional categories:
- **Category A:** Natural Language Inference (NLI) & Semantic Reasoning
- **Category B:** Computational Dependency Detection & Causal Provenance
- **Category C:** Evidence Verification & Fact Checking
- **Category D:** Specialist Agent Reasoning (Insurance Policy, Risk, Documents)
- **Category E:** Fault, Corruption, and Anomaly Detection
- **Category F:** Provenance and Recovery Testing (Deterministic Graph Invariants)

---

## 2. Category F Scientific Determination: ML vs. Deterministic Testing

> [!IMPORTANT]
> **Definitive Research Determination on Category F:**  
> **Category F (Provenance Tracking and Selective Recovery) MUST NOT use a trained machine learning model.**  
> It must be evaluated exclusively through a **deterministic graph and cryptographic test harness**.

### Scientific Justification:
1. **Mathematical Invariant Guarantees:** Causal provenance in this system is defined by a Directed Acyclic Graph (DAG) whose edges are immutable SHA-256 parent hashes. Whether a node $C$ is a causal descendant of node $A$ is an exact graph-reachability problem ($\text{Reach}(A, C)$), computable in $O(V + E)$ via Breadth-First Search (BFS). Introducing a machine learning classifier to predict reachability would introduce non-zero false negatives (failing to invalidate stale certificates) and false positives (needlessly discarding valid work).
2. **Cryptographic Security & Non-Repudiation:** Cryptographic tip commitment and anchor verification rely on SHA-256 digest integrity. Machine learning models are inherently approximate and cannot verify bitwise cryptographic integrity.
3. **Auditability in Adjudication:** In financial, legal, and medical adjudication, selective recovery decisions must be auditable in court or regulatory review. Deterministic graph traversal provides a 100% explainable, mathematically provable lineage.

Therefore, Category F is tested via the deterministic harness in `tests/test_consensus_recovery.py` and `src/cert_recovery/engine.py`.

---

## 3. Audit of Existing Repository Datasets

The repository currently contains 5 primary dataset artifacts:

| Dataset Name | Repository Path | Size / Rows | Input Schema & Format | Available Labels | Project Role | Limitations, Biases & Confounds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`baseline_tests.json`** | `baseline_tests.json` | 6 pairs | JSON array: `{"id", "premise", "hypothesis", "expected_label"}` | `entailment`, `contradiction`, `neutral` | Pinned smoke test for NLI cross-encoder models (`MiniLM`, `DeBERTa`) | Microscopic sample size (6 items); 1 test (`nli-003`) fails on both MiniLM and DeBERTa due to lexical dissimilarity treated as contradiction. |
| **`Synthetic Pilot Cases`** | `data/synthetic_pilot_cases.jsonl` | 200 cases | JSON Lines: Claim metadata, policy clauses, invoices, police reports | Adjudication decisions (`APPROVE`, `REJECT`, `REVIEW`) | End-to-end multi-agent adjudication validation | Fully synthetic; lack lexical nuances of real-world scanned medical documents or unstructured adjustor notes. |
| **`Candidate V2 Controlled Suite`** | `_private/runs/candidate_v2_*/controlled_data.jsonl` | 568 total rows (256 train, 64 val, 128 stress, 120 final) | JSON Lines: `{"id", "source_artifact", "candidate_artifact", "label", "split", "category"}` | Binary: `0` (`independent`), `1` (`depends_on`) | Supervised training and evaluation of `DeBERTa-v3-small` dependency classifier | **Severe Semantic Confound:** Contains 16 `nli_context` test rows where the hypothesis repeats factual statements from candidate artifacts. The model learned to predict dependency based on lexical/semantic truth rather than declared DAG support relationships. |
| **`Candidate V3 Root-Explicit Contrast`** | `docs/DEPENDENCY_TARGET.md` & `src/cert_recovery/v3_preparation.py` | 512 total rows (256 train, 64 val, 128 future-final) | Structured text pairs with explicit ancestral closure clause: *"All sources have no supports."* | Binary: `0` (`independent`), `1` (`depends_on`) | Designed to eliminate the NLI confound by training models to attend to explicit root provenance rules | **Status: PREPARED_NOT_RUN.** Dataset is synthesized and sealed. Prevents benchmark leakage, but has not yet been experimentally trained or validated. |
| **`Factorial Ablation Scenarios`** | `_private/notebooks/Research_Evaluation.ipynb` | 150 trials (9 arms × 3 seeds × 5 cases) | Parameterized scenario configurations across policy, docs, fraud, and consent | Adjudication decisions, unsafe approvals, review transitions | Empirical measurement of Rule Gate, Specialist Composition, and Selective Recovery | Synthetic scenario permutations designed specifically to stress test corner cases; not suitable for gradient-based training. |

---

## 4. Proposed External Research Benchmark Datasets

To establish a defensible, publication-grade benchmark comparing our 12 models, the following peer-reviewed, open-access datasets are selected:

### Category A: NLI and Semantic Reasoning
1. **Multi-Genre Natural Language Inference (MultiNLI)**
   - **Source:** Williams et al. (NAACL 2018), New York University.  
   - **URL:** [https://huggingface.co/datasets/nyu-mll/multi_nli](https://huggingface.co/datasets/nyu-mll/multi_nli)  
   - **Size:** 433,000 sentence pairs across 10 genres.  
   - **Labels:** `entailment`, `contradiction`, `neutral`.  
   - **License:** ODC Public Domain Dedication and Licence (PDDL).  
   - **Project Relevance:** Evaluates general cross-encoder semantic robustness and prevents out-of-domain degradation.
2. **ContractNLI**
   - **Source:** Koreeda & Manning (EMNLP 2021), Stanford University.  
   - **URL:** [https://huggingface.co/datasets/koreeda/contract_nli](https://huggingface.co/datasets/koreeda/contract_nli)  
   - **Size:** 607 non-disclosure contracts, 10,128 clause-hypothesis pairs.  
   - **Labels:** `entailment`, `contradiction`, `not_mentioned` (neutral).  
   - **License:** CC BY-NC-SA 4.0.  
   - **Project Relevance:** Highly specific to legal agreements and insurance policy clauses. Tests whether models can determine if a specific claim is supported or prohibited by contractual policy wording.

---

### Category B: Computational and Causal Dependency Detection
1. **Extended Repository Provenance Suite (Candidate V2 + Candidate V3 Contrast)**
   - **Source:** Repository authored (`controlled_data.py`, `v3_preparation.py`).  
   - **Size:** 1,080 annotated provenance pairs (512 train, 128 val, 248 stress, 192 held-out test).  
   - **Labels:** `0` (`independent`), `1` (`depends_on`).  
   - **Structure:** `[Premise / Upstream Artifact Content + Ancestry Declaration]` vs `[Hypothesis / Downstream Assessment Claim]`.  
   - **Project Relevance:** The only ground-truth dataset directly reflecting our cryptographic DAG dependency semantics.

---

### Category C: Evidence Verification and Fact Checking
1. **Fact Extraction and VERification (FEVER)**
   - **Source:** Thorne et al. (NAACL 2018), University of Sheffield / Amazon.  
   - **URL:** [https://huggingface.co/datasets/fever/fever](https://huggingface.co/datasets/fever/fever)  
   - **Size:** 185,445 claims annotated with supporting/refuting Wikipedia evidence.  
   - **Labels:** `SUPPORTS`, `REFUTES`, `NOT_ENOUGH_INFO`.  
   - **License:** CC BY-SA 3.0.  
   - **Project Relevance:** Directly mirrors our specialist agent evidence verification task: determining whether an assessment's rationale is genuinely supported by the attached medical invoice or police report.

---

### Category D: Specialist Agent Reasoning (Policy, Claims, Compliance)
1. **CUAD (Contract Understanding Atticus Dataset)**
   - **Source:** Hendrycks et al. (NeurIPS 2021), Atticus Project / UC Berkeley.  
   - **URL:** [https://huggingface.co/datasets/the_atticus_project/cuad](https://huggingface.co/datasets/the_atticus_project/cuad)  
   - **Size:** 510 contracts with 13,101 expert annotations across 41 legal categories.  
   - **Structure:** Context (contract text), Question (policy category clause), Answer span.  
   - **License:** CC BY 4.0.  
   - **Project Relevance:** Benchmarks LLMs on extracting binding terms (e.g., deductible limits, expiration clauses, territorial restrictions).
2. **InsuranceQA (Corpus V2)**
   - **Source:** Shen et al. (2015), InsuranceQA.  
   - **URL:** [https://github.com/shuzi/insuranceQA](https://github.com/shuzi/insuranceQA)  
   - **Size:** 26,000 real-world insurance questions with professional adjustor answers.  
   - **Structure:** Question, domain label, candidate answer pools.  
   - **License:** Apache License 2.0.  
   - **Project Relevance:** Evaluates generative domain fluency and claim assessment generation for insurance specialist agents.

---

### Category E: Fault, Corruption, and Hallucination Detection
1. **HaluEval (Hallucination Evaluation for LLMs)**
   - **Source:** Li et al. (EMNLP 2023), Renmin University of China.  
   - **URL:** [https://github.com/RUCAIBox/HaluEval](https://github.com/RUCAIBox/HaluEval)  
   - **Size:** 35,000 document-grounded generations with paired factual vs hallucinated responses.  
   - **Labels:** `factual`, `hallucinated`.  
   - **License:** MIT License.  
   - **Project Relevance:** Benchmarks our guardrail and fault detection models (`Llama-Guard-3-1B` and DeBERTa FEVER) on identifying unsupported agent proposals before consensus insertion.

---

## 5. Dataset to Model Compatibility Matrix

The table below maps each shortlisted model to its target research task, primary evaluation dataset, and supervised training dataset:

| Model Candidate | Target Research Task | Evaluation Dataset | Supervised Training Dataset | Input Format | Output / Target Label |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. MiniLM2-L6-H768** | Semantic Advice / NLI Baseline | MultiNLI + ContractNLI + `baseline_tests.json` | Pretrained (Inference comparator) | Sequence pair: `[CLS] Premise [SEP] Hypothesis [SEP]` | 3-class logits (`entailment`, `neutral`, `contradiction`) |
| **2. DeBERTa-v3-small** | Pinned Semantic NLI Baseline | MultiNLI + ContractNLI + `baseline_tests.json` | Pretrained / Fine-tuned on ContractNLI | Sequence pair: `[CLS] Premise [SEP] Hypothesis [SEP]` | 3-class logits |
| **3. ModernBERT-base-nli** | Long-Context Document NLI | ContractNLI (Full contracts up to 8k tokens) | MultiNLI + ContractNLI | Long-sequence pair (up to 8,192 tokens) with unpadding | 3-class logits |
| **4. TF-IDF + Logistic Reg** | Linear Dependency Baseline | Candidate V2 & V3 Provenance Suites | Candidate V2 & V3 Train Split (256 rows) | N-gram sparse vector of concatenated artifact texts | Binary probability: $P(\text{depends\_on} \in [0, 1])$ |
| **5. DeBERTa-v3-small-dep** | Neural Dependency Classifier | Candidate V2 & V3 Provenance Suites | Candidate V3 Root-Explicit Contrast Train Split (256 rows) | Sequence pair: `Artifact A text + Ancestry` vs `Artifact B text` | Binary classification (`0`: independent, `1`: depends) |
| **6. BGE-small-en-v1.5** | Dense Embedding Baseline | Candidate V2 & V3 Provenance Suites | Unsupervised / Cosine similarity threshold tuning | Independent sequences: `Embed(Artifact A)` and `Embed(Artifact B)` | Continuous cosine similarity $\in [-1, 1]$ mapped to threshold |
| **7. Rule Engine Oracle** | Ground-Truth Invariant Oracle | Extended Factorial Ablation Scenarios (150 trials) | None (Deterministic symbolic logic) | Structured claim JSON schema | Exact deterministic boolean / decision (`APPROVE`, `REJECT`, `REVIEW`) |
| **8. DeBERTa-v3-base-FEVER** | Claim Evidence Verification | FEVER test set + Claim-Invoice Evidence Pairs | FEVER train split + ContractNLI | Sequence pair: `[Document Evidence]` vs `[Agent Claim Rationale]` | 3-class logits (`SUPPORTS`, `REFUTES`, `NOT_ENOUGH_INFO`) |
| **9. Llama-Guard-3-1B** | Agent Fault & Guardrail Classifier | HaluEval QA + Synthetic Corrupted Agent Outputs | Instruction-tuned with LoRA on Agent Audit Split | Prompt template: `[Agent Rationale + Claim Data]` | Structured safety token: `safe` or `unsafe [violation_code]` |
| **10. Qwen2.5-0.5B-Instruct** | Compact Generative Specialist | 50 Annotated Insurance Claim Scenarios | InsuranceQA + Synthetic Claim Pilot (2-shot SFT) | ChatML prompt with 2-shot exemplar and claim document | Valid JSON schema: `{"decision", "confidence", "rationale", "flags"}` |
| **11. Llama-3.2-1B-Instruct** | 1B-Tier Generative Specialist | 50 Annotated Insurance Claim Scenarios | InsuranceQA + CUAD Contract SFT (LoRA) | Llama-3 Chat template with system instruction and claim text | Valid JSON schema with strict evidence citing |
| **12. Phi-3.5-mini-instruct** | Lab Frontier Reasoning Benchmark | 50 Annotated Insurance Claim Scenarios | SFT + DPO on Multi-Specialist Claim Reasoning | Chat prompt with multi-page policy context | Complex multi-clause JSON assessment with audit traces |

---

## 6. Data Integrity and Leakage Prevention Protocols

To ensure scientific defensibility:
1. **Strict Hash-Partitioned Splits:** All splits across Candidate V2, Candidate V3, and domain claims are partitioned by hash of the claim ID (`hash(claim_id) % 100`). This ensures that no candidate artifact from the training split ever appears in the validation, stress, or final test splits.
2. **Decoupled Causal Provenance from NLI Entailment:** The Candidate V3 dataset incorporates explicit counter-factual cases: pairs where statements are 100% semantically entailed in the real world but have *zero declared causal dependency* in the system's DAG.
3. **Sealed Final Test Sets:** Final evaluation splits remain sealed in `_private/` and are evaluated strictly once after model hyperparameters and thresholds are frozen on validation splits.
