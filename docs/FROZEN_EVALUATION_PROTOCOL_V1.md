# Frozen Phase 5/6 Evaluation Protocol (`protocol-v1`)

**Specification:** [FROZEN_DECISION_SPEC_V1.md](FROZEN_DECISION_SPEC_V1.md) (`spec-v1`)  
**Frozen Manifest:** `results/phase5/frozen_evaluation_manifest.json`  
**Phase 5 Rule:** Specification, distinct splits, oracle labels, baselines, statistical rules, and live Groq budget are frozen here; NO system or baseline is executed on the held-out `test` split until Phase 6.

---

## 1. Dropped Historical Strawman & Primary Evaluation Arms

**Explicitly Dropped Strawman:** The exclusion-blind legacy prototype (`Arm E / mode="legacy"` from `v0-paper`, whole-claim weighted role voting without substantive exclusion authority) is **dropped as a primary comparator** and retained strictly as a deprecated historical reference note.

### Primary Comparison Arms (Phase 6)
1. **`FULL_DET` (Deterministic Full System):** `SpecialistCompositionAdjudication` (`composition-v1`, `substantive_exclusion`), deterministic `RuleAgent` per role, Phase 2 `verify_composition_independently`, Phase 3 Ed25519 signed `EventDAG`, and `CertificateSystem` selective recovery.
2. **`FULL_LIVE` (Live Model Full System):** Identical architecture with all 4 roles backed by **separate role-isolated calls** to `groq:openai/gpt-oss-20b` (`>=7B` policy compliant, `temperature=0.0`, `min_call_interval_sec=5.0`).
3. **`MAJORITY_VOTE`:** Unweighted plurality/majority across the 4 specialist role decisions without mandatory exclusion gate (ties -> `REVIEW`).
4. **`CONFIDENCE_WEIGHTED_VOTE`:** Confidence-weighted sum across specialist decisions without mandatory exclusion gate.
5. **`MAJORITY_POLICY_VETO` (Strong Non-Strawman Baseline):** Respects validated `policy` exclusions (`REJECT` if `policy` reports valid `REJECT`), otherwise takes unweighted majority of specialist role outputs.
6. **`SINGLE_AGENT_E2E_LLM`:** Single `groq:openai/gpt-oss-20b` call seeing all 10 claim fields jointly and returning one whole-claim decision (`APPROVE`/`REJECT`/`REVIEW`), bypassing role isolation and typed obligation decomposition.
7. **`LLM_AS_JUDGE`:** Separate `groq:openai/gpt-oss-20b` judge call reviewing the 4 specialist outputs and choosing the final verdict without deterministic `composition-v1` gate enforcement (planned on the stratified budget-capped live subset or reusing `FULL_LIVE` specialist outputs).

### Five Targeted Phase 6 Ablations
| Ablation ID | Change Relative to `FULL_DET` / `FULL_LIVE` | Primary Metric Isolated |
| :--- | :--- | :--- |
| `no_gate` | Bypass `composition-v1` mandatory policy-exclusion gate | Unsafe-Approval Rate (`unsafe_approval_rate`) |
| `no_validation` | Disable `assess_specialist` schema/binding checks | Malformed & invalid-output containment rate |
| `no_certificates` | Disable `CertificateSystem` staleness/hash binding checks | Stale-decision rate on `v1 -> v2` claim updates |
| `no_provenance` | Disable Ed25519 `EventDAG` signing & checkpointing | Tamper/rollback detection & lineage auditability |
| `no_recovery` | Full clean re-run on claim update instead of selective rebuild | Calls, total tokens, latency, and output equivalence |

---

## 2. Exact Metric Definitions & Denominators

Let $N$ be total evaluated cases, with oracle label $y_i \in \{\text{APPROVE}, \text{REJECT}, \text{REVIEW}\}$ and system output $\hat{y}_i$:

- **Unsafe-Approval Rate:**  
  $$\text{UnsafeApprovalRate} = \frac{\#\{i : \hat{y}_i = \text{APPROVE} \land y_i \neq \text{APPROVE}\}}{\#\{i : y_i \neq \text{APPROVE}\}}$$  
  *(Also report raw count over $N$ alongside this conditioned denominator.)*
- **False-Reject Rate:**  
  $$\text{FalseRejectRate} = \frac{\#\{i : \hat{y}_i = \text{REJECT} \land y_i \neq \text{REJECT}\}}{\#\{i : y_i \neq \text{REJECT}\}}$$
- **Coverage:** $\frac{\#\{i : \hat{y}_i \in \{\text{APPROVE}, \text{REJECT}\}\}}{N}$
- **Review (Abstention) Rate:** $\frac{\#\{i : \hat{y}_i = \text{REVIEW}\}}{N}$
- **Selective Accuracy (Accuracy at Coverage):** $\frac{\#\{i : \hat{y}_i \in \{\text{APPROVE}, \text{REJECT}\} \land \hat{y}_i = y_i\}}{\#\{i : \hat{y}_i \in \{\text{APPROVE}, \text{REJECT}\}\}}$
- **Overall Exact-Match Accuracy:** $\frac{\#\{i : \hat{y}_i = y_i\}}{N}$
- **Cost & Recovery Metrics:** First-run API calls, post-update API calls, prompt/completion/total tokens, end-to-end latency (ms), output equivalence rate vs. full re-run, and stale-decision rate.

---

## 3. Statistical Protocol

1. **Multi-seed offline sweeps:** `10` independent seeds for deterministic & stochastic fault-injection experiments (`[101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]`).
2. **Confidence intervals:** `95%` **Wilson score intervals** for binomial proportions AND **family-clustered / paired case-level bootstrap CIs** (`B = 2000` resamples) accounting for template-family clustering.
3. **Hypothesis testing:** **Exact two-sided McNemar's test** strictly on paired binary outcomes with one independent decision per case (e.g., exact-match correct vs. incorrect, or safe vs. unsafe on non-approve cases). Multi-seed pooled tables and subgroup breakdowns are explicitly labeled **exploratory**.

---

## 4. Calibrated Groq `8,000 TPM` Execution Budget (`openai/gpt-oss-20b`)

**Phase 4 Measured Calibration (`results/phase4/pilot_summary.json` & `live_pilot_calls.jsonl`):**
- Clean live dev calls (`24` calls): `525.54` avg total tokens/call (`359.12` prompt + `166.42` completion; max `657`), avg latency `4.73s/call`.
- Combined Phase 4 dev + fault calls (`58` calls): `462.57` avg total tokens/call (`26,829` total tokens), `$0.00` free-tier USD spend, `min_call_interval_sec = 5.0` (`~12 calls/min` = `~6,306 TPM` < `8,000 TPM` free-tier ceiling).
- Conservative planning rate for Phase 6 budgeting: **`550` tokens/call** at **`5.0s/call`** (`12 calls/min`).

See `results/phase5/frozen_evaluation_manifest.json` for exact Option A (`104`-claim primary track) vs. Option B (`36`-claim stratified multi-arm live subset if Groq RPD/TPD daily limits constrain multi-arm live runs) budgets and stopping rules.
