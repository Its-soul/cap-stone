# Verifiable Multi-Agent Fault-Tolerant Adjudication System with Cryptographic Causal Provenance

An academic research system combining a **deterministic versioned certificate engine and SHA-256 provenance DAG** with **advisory machine learning models** for fault-tolerant multi-agent decision workflows.

---

## START HERE

- **Live Browser Demonstration:** Run `python scripts/run_claim_demo.py` from the repository root and open `http://127.0.0.1:8000/` (frontend source: [demo/index.html](demo/index.html)).
- **Core Implementation:** [src/cert_recovery/engine.py](src/cert_recovery/engine.py) (DAG & certificate recovery) and [src/cert_recovery/adjudication/workflow.py](src/cert_recovery/adjudication/workflow.py) (multi-agent adjudication).
- **Offline Validation Notebook:** [notebooks/01_Adjudication_Demo.ipynb](notebooks/01_Adjudication_Demo.ipynb).
- **Candidate V2 Dependency Model Walkthrough:** [notebooks/03_Dependency_Model_V2.ipynb](notebooks/03_Dependency_Model_V2.ipynb).
- **Verified Empirical Results:** [docs/EVALUATION.md](docs/EVALUATION.md) and [results/](results/).

---

## Problem Statement and Core Contribution

Multi-agent LLM decision pipelines are vulnerable to silent hallucinations, cascading corruption across agents, and expensive full-pipeline restarts whenever a single source document changes.

This project enforces a **strict boundary between untrusted advisory AI/ML models and authoritative deterministic verification**:

1. **Disjoint Specialist Composition (`composition-v1`):** `policy`, `evidence`, `risk`, and `compliance` specialists evaluate domain-specific obligations (`coverage-v1`, `documents-v1`, `fraud-screen-v1`, `identity-consent-v1`) rather than casting interchangeable whole-claim votes.
2. **Non-Overridable Mandatory Safety Rules:** Deterministic rule gates and typed schema validation intercept malformed, conflicting, or policy-ineligible proposals (`0` unsafe approvals across verified ablation suites). Confidence scores have zero certificate or payout authority.
3. **SHA-256 Causal Provenance & Selective Recovery:** Every artifact and certificate transition is hash-chained in `EventDAG` (`src/cert_recovery/adjudication/ledger.py`). When an evidence fact changes, only affected downstream nodes are invalidated and recomputed while unaffected specialist certificates remain valid at version `v1`.
4. **Advisory Dependency Prediction:** A fine-tuned `DeBERTa-v3-small` model (`Candidate V2`) predicts candidate dependency edges without modifying authoritative DAG edges or bypassing verification.

---

## System Workflow

```text
Claim Evidence (10 domain facts)
      |
      +--> Policy      (coverage, waiting period, limit)  --+
      +--> Evidence    (documents available, invoice match) |
      +--> Risk        (duplicate claim, provider flag)     +--> Schema & Rule Validation
      +--> Compliance  (identity verified, consent)       --+
                                                            |
                                  +-------------------------+
                                  v
            Specialist Composition & Mandatory Rule Gate
             - Policy violated        -> REJECT
             - Any failure/uncertain  -> REVIEW
             - All checks satisfied   -> APPROVE
                                  |
                                  v
            Versioned Certificate & SHA-256 EventDAG
             - Edit 1 fact -> Invalidate affected branch only
             - Reuse unaffected v1 certificates -> Update decision
```

### Selective Recovery Example

1. A hospitalization claim is submitted with `documents_available = No` (`policy`, `risk`, and `compliance` pass).
2. All four specialists run (`v1`), and the system returns **REVIEW** (required documents missing).
3. The claimant supplies the missing document (`documents_available` changes `No -> Yes`).
4. `CertificateSystem` invalidates only the `evidence` branch: `evidence` reruns (`Reran`), while `policy`, `risk`, and `compliance` remain **Reused** at `v1`, issuing round certificate `v2` and updating the decision to **APPROVE**.

---

## Quickstart

### Installation

Requires **Python 3.11+**. The core engine requires only `PyYAML`.

```sh
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux:        source .venv/bin/activate
python -m pip install -e ".[remote,test]"
```

Optional ML extras (`laptop`, `nvidia`) are declared in [pyproject.toml](pyproject.toml) and should be kept in a separate environment from `remote` due to `huggingface-hub` version constraints (see [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)).

### Run the Demonstration

```sh
python scripts/run_claim_demo.py
```

Open `http://127.0.0.1:8000/`. Use **Run claim** to evaluate a case, edit any claim field and click **Apply changes** to observe selective recomputation and audit events, or click **New claim** to reset. Pass `--port <PORT>` to change the port. The optional V2 dependency panel loads a local `CERT_RECOVERY_V2_RUN` bundle when launched from an environment with `torch` and `transformers`.

### Run Tests and Offline Validation

```sh
# Complete offline test suite
python -m pytest tests -q

# End-to-end deterministic prototype, tamper detection, and recovery demo
python scripts/run_insurance_prototype.py

# Validate public configs, pins, documentation links, and notebook syntax
python scripts/validate_preparation.py
python scripts/validate_notebooks.py
```

---

## Repository Structure

```text
certificate-recovery/
|-- README.md
|-- pyproject.toml
|-- requirements.txt
|-- requirements-colab.txt
|-- .env.example
|-- .gitignore
|-- run_baseline.py              # Optional remote NLI smoke runner
|-- remote_model_client.py       # Public HF Space adapter client
|-- semantic_advice.py           # Version/hash-bound advisory NLI helper
|-- baseline_tests.json          # Six diagnostic NLI smoke fixtures
|-- config/                      # System, pinned model, and benchmark configs
|-- data/examples/               # Reference JSONL schema examples
|-- demo/                        # Interactive local browser demonstration (index.html)
|-- docs/                        # Consolidated technical documentation
|-- notebooks/                   # Five canonical research and demo notebooks
|-- research_benchmark/          # 12-model registry, loaders, trainers, metrics, plots
|-- scripts/                     # CLI demos, ablation runners, report builders, validators
|-- src/cert_recovery/           # Core engine, crypto DAG, adjudication, and ML pipelines
|-- tests/                       # Unit, adjudication, integration, ML, and remote tests
|-- results/                     # Ignored generated metrics, reports, and figures
`-- _private/                    # Ignored frozen checkpoints, logs, and internal archive
```

---

## Verified Results Summary

*All offline prototype benchmarks use deterministic rule simulators or synthetic fixtures unless explicitly noted as local checkpoint or ONNX inference.*

- **Factorial Adjudication Ablation (`N = 150`, seeds `42, 101, 2024`):** Enabling the deterministic rule gate improved 3-class accuracy by **+48.17 pp** on average across matched configurations (`Arm E`: 72.0% vs. `Arm D`: 22.7%) and eliminated unsafe approvals (`0` across all gated arms vs. `107–115` without gating).
- **Specialist Composition (`composition-v1`):** Replacing interchangeable whole-claim votes with disjoint specialist obligations eliminated the 42 false `REVIEW` deferrals observed under legacy voting while preserving `0` unsafe approvals.
- **Selective Recovery Benchmark (`results/recovery/`):** Single-branch updates rebuild `1` specialist while preserving `3` at `v1` (`4.0` vs. `7.0` configured cost units). Under ~0.01ms in-memory mock agents, graph hashing limits wall-clock speedup to `~1.12x–1.27x` (and can be slower under noise); meaningful wall-clock savings appear when avoiding heavy LLM inference calls (~37s/call on local CPU for INT8 `Qwen2.5-0.5B`).
- **Candidate V2 Dependency Model (`DeBERTa-v3-small`, $\tau = 0.20$):** Achieved **87.5%** validation accuracy (`0.873` macro-F1, `n=64`), **91.7%** final held-out accuracy (`0.916` macro-F1, `0` false negatives, `n=96`), and **82.8%** stress accuracy (`n=384`), while exposing a semantic-overlap confound (`16/16` failures on `nli_context` independent rows) that motivated the **unexecuted** Candidate V3 contrast formulation.
- **Compliance Model Adapter:** Schema enforcement and `CONFIRMED_FAILURE` weight zeroing were verified across 11 cases via `mock_compliance_backend` (`live_backend_executed: false`); prompt calibration on `Qwen2.5-0.5B-Instruct` ONNX was evaluated separately on 4 scenarios.

---

## Known Limitations

- **Simulation vs. Live Multi-LLM Inference:** End-to-end adjudication suites rely on deterministic `RuleAgent` simulators and mock compliance outputs rather than live multi-GPU LLM execution.
- **Uncalibrated Confidence:** Agent confidence values are authored or uncalibrated; probability calibration (Platt/temperature scaling) remains future work.
- **Cryptographic Trust Boundary:** `EventDAG` uses unkeyed SHA-256 hashes; detecting history truncation requires an independently retained checkpoint anchor, without asymmetric signatures or formal Byzantine quorum protocols.
- **40% Adversarial Claim Unproven:** Synthetic Bernoulli fault-injection tests at requested rates up to 40% illustrate rule containment on small authored suites and do **not** establish general Byzantine fault tolerance.
- **Candidate V3 Prepared Only:** V3 datasets and launch guards exist in `src/cert_recovery/v3_preparation.py` and `scripts/run_candidate_v3.py`, but remain intentionally unexecuted pending external review.

---

## Documentation

- [Architecture & Data Contracts](docs/ARCHITECTURE.md)
- [Research Methodology & Dependency Target](docs/METHODOLOGY.md)
- [Models, Baselines & Execution Tiers](docs/MODELS.md)
- [Evaluation & Reconciled Findings](docs/EVALUATION.md)
- [Reproducibility & Verification Guide](docs/REPRODUCIBILITY.md)
