# Evaluation and Reconciled Findings

All metrics below are strictly verified against saved experimental logs (`results/` and `_private/reconciliation_20261012/`). Prepared/unexecuted pipelines are labeled explicitly and never mixed with measured results.

## 1. Multi-Agent Adjudication Ablation (Extended Suite, N = 150)

Across 10 synthetic insurance scenarios (`2 APPROVE`, `3 REJECT`, `5 REVIEW`), 5 Bernoulli fault rates (`0.0..0.4`), and 3 seeds (`42, 101, 2024`), the 9-arm factorial study tests `(Majority vs. Confidence-Weighted) x (Validation OFF vs. ON) x (Mandatory Gate OFF vs. ON)` plus the deterministic rule oracle (`Arm F`):

| Arm | Method | 3-Class Accuracy | Decisive Coverage | Decided Accuracy | Unsafe Approvals | Additional Deferrals |
|---|---|---:|---:|---:|---:|---:|
| **A** | Majority only | 22.7% (34/150) | 93.3% (140/150) | 22.1% (31/140) | 107 | 7 |
| **B** | Majority + Gate | 70.7% (106/150) | 20.7% (31/150) | 100.0% (31/31) | **0** | 44 |
| **C** | Weighting only | 23.3% (35/150) | 100.0% (150/150) | 23.3% (35/150) | 109 | 0 |
| **D** | Weighting + Validation (no gate) | 22.7% (34/150) | 98.7% (148/150) | 22.3% (33/148) | 115 | 1 |
| **E** | Full Legacy Prototype (Weight + Val + Gate) | 72.0% (108/150) | 22.0% (33/150) | 100.0% (33/33) | **0** | 42 |
| **F** | Deterministic Rules Oracle | **100.0% (150/150)** | **50.0% (75/150)** | **100.0% (75/75)** | **0** | **0** |
| **G** | Majority + Validation + Gate | 70.7% (106/150) | 20.7% (31/150) | 100.0% (31/31) | **0** | 44 |
| **H** | Majority + Validation (no gate) | 25.3% (38/150) | 93.3% (140/150) | 22.1% (31/140) | 109 | 3 |
| **I** | Weighting + Gate (no validation) | 73.3% (110/150) | 23.3% (35/150) | 100.0% (35/35) | **0** | 40 |

### Key Mechanism Findings

- **Mandatory Rule Gate Is Essential:** Enabling the gate adds **+48.17% mean 3-class accuracy** across matched pairs (`B-A`, `G-H`, `I-C`, `E-D`) and reduces unsafe approvals from 107–115 down to **0**.
- **Confidence Weighting Alone Adds +1.33 pp:** Comparing `Arm E` (72.0%) vs. `Arm G` (70.7%) isolates confidence weighting under identical validation and gating. Without the gate (`Arm C`, `Arm D`), confidence weighting produced 109–115 unsafe approvals.
- **Role-Fusion Category Error (`Arm E` vs. `Arm F`):** In legacy voting, non-policy agents voted `APPROVE` on policy exclusion claims (3-to-1 `APPROVE` vs. `REJECT`), causing the gate to defer 42 valid rejections to `REVIEW`.
- **Composition Fix (`composition-v1`):** Evaluated via `scripts/run_composition_evaluation.py` (`results/composition/`), replacing interchangeable whole-claim votes with disjoint specialist obligations (`SpecialistCompositionAdjudication`) eliminates those false deferrals while preserving 0 unsafe approvals.

## 2. Selective Recovery Benchmark

Single-pass full recomputation (strictly 4 agent executions) vs. selective DAG recovery (`src/cert_recovery/adjudication/recovery_benchmark.py`, saved in `results/recovery/recovery_benchmark_repaired_results.json`):

| Scenario | Rebuilt (Selective) | Rebuilt (Full) | Preserved (`v1`) | Selective Cost | Full Cost | Saved Executions |
|---|---:|---:|---:|---:|---:|---:|
| Single agent failure (`risk`) | 1 | 4 | 3 | 4.0 | 7.0 | 3 |
| Single fact change (`evidence`) | 1 | 4 | 3 | 4.0 | 7.0 | 3 |
| Multi-branch change (`policy + evidence`) | 2 | 4 | 2 | 5.0 | 7.0 | 2 |
| Repeated changes before recovery | 2 | 4 | 2 | 5.0 | 7.0 | 2 |

**Timing Disclosure:** With instantaneous in-memory Python mock agents (~0.01ms/call), wall-clock speedup is modest (~1.12x–1.27x) and can drop below 1.0x under system noise because DAG traversal and SHA-256 hashing dominate runtime. Selective recovery yields major wall-clock savings only when agent inference is expensive (e.g., skipping 3 local ~37s LLM calls saves ~110s per update).

## 3. Candidate V2 & Compliance Adapter Verification

- **Candidate V2 (`DeBERTa-v3-small` dependency head):** Validation (`n=64`, $\tau=0.20$): 87.5% acc / 0.873 macro-F1; Held-out final (`n=96`): 91.7% acc / 0.916 macro-F1 (0 FN); Stress (`n=384`): 82.8% acc / 0.823 macro-F1 (42 improvements, 49 regressions). Known confound: 16/16 `nli_context` independent rows misclassified as dependent.
- **Compliance Model Adapter (`results/model_agent/model_evaluation_results.json`):** Verified schema parsing, `CONFIRMED_FAILURE` weight zeroing (for malformed JSON/timeouts), and containment across 11 compliance cases using `mock_compliance_backend` (`live_backend_executed: false`). Calibration scaling and live multi-agent LLM inference remain unexecuted.
- **Historical & Statistical Protocol (`research_benchmark/evaluation.py`):** Paired group comparisons use McNemar's test, bootstrap confidence intervals, and False Negative Rate ($\text{FNR}_{\text{dep}}$) prioritization to ensure stale certificates are never silently reused.
