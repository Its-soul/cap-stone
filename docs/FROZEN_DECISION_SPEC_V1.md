# Frozen Decision and Obligation Specification (`spec-v1`)

**Version:** `spec-v1`  
**Status:** FROZEN FOR PHASE 5/6 EVALUATION  
**Domain:** Synthetic Healthcare Insurance Claim Adjudication (`SYNTHETIC_PROVISIONAL`)

---

## 1. Canonical Claim Input Schema

Every claim consists of a unique `claim_id` and four disjoint specialist fact slices (enforced by runtime input scoping):

| Role | Contract Version | Obligation Name | Scoped Fact Fields & Types |
| :--- | :--- | :--- | :--- |
| `policy` | `coverage-v1` | `policy_eligibility` | `amount_inr` (`int > 0`), `coverage_limit_inr` (`int > 0`), `hospitalization_covered` (`bool`), `waiting_period_satisfied` (`bool`) |
| `evidence` | `documents-v1` | `evidence_integrity` | `documents_available` (`bool`), `invoice_matches` (`bool`) |
| `risk` | `fraud-screen-v1` | `screening_integrity` | `duplicate_claim` (`bool`), `provider_flagged` (`bool`) |
| `compliance` | `identity-consent-v1` | `regulatory_controls` | `identity_verified` (`bool`), `consent_available` (`bool`) |

**Invariants:**
- All integer amounts must be strictly positive (`type(x) is int and x > 0`).
- All boolean predicates must be explicit `bool` values (`type(x) is bool`).
- No runtime specialist receives fields outside its own role row, nor any oracle/expected label.

---

## 2. Role Obligation & Consequence Semantics (`SAT` / `FAIL` / `UNK`)

Each specialist output is mapped by the host validator (`assess_specialist`) and independently re-derived by `independent_verifier` against authoritative source artifacts:

### 2.1 Truth Evaluation per Role (When Structural & Binding Checks Pass)
1. **`policy` (`coverage-v1` -> `policy_eligibility`):**
   - `SAT` (consequence `APPROVE`) iff `hospitalization_covered == True` AND `waiting_period_satisfied == True` AND `amount_inr <= coverage_limit_inr`.
   - `FAIL` (consequence `REJECT`) otherwise (substantive coverage exclusion or limit breach).
   - **Boundary:** Exact limit equality (`amount_inr == coverage_limit_inr`) evaluates to `SAT` (`APPROVE`). Exceeding by 1 INR (`amount_inr == coverage_limit_inr + 1`) evaluates to `FAIL` (`REJECT`).
   - Permitted role consequences: `{"APPROVE", "REJECT"}`. `REVIEW` is an impossible role consequence for `policy`.
2. **`evidence` (`documents-v1` -> `evidence_integrity`):**
   - `SAT` (consequence `APPROVE`) iff `documents_available == True` AND `invoice_matches == True`.
   - `FAIL` (consequence `REVIEW`) otherwise (procedural evidence deficiency).
   - Permitted role consequences: `{"APPROVE", "REVIEW"}`. `REJECT` is an impossible role consequence for `evidence`.
3. **`risk` (`fraud-screen-v1` -> `screening_integrity`):**
   - `SAT` (consequence `APPROVE`) iff `duplicate_claim == False` AND `provider_flagged == False`.
   - `FAIL` (consequence `REVIEW`) otherwise (procedural fraud-screening trigger).
   - Permitted role consequences: `{"APPROVE", "REVIEW"}`. `REJECT` is an impossible role consequence for `risk`.
4. **`compliance` (`identity-consent-v1` -> `regulatory_controls`):**
   - `SAT` (consequence `APPROVE`) iff `identity_verified == True` AND `consent_available == True`.
   - `FAIL` (consequence `REVIEW`) otherwise (procedural regulatory gap).
   - Permitted role consequences: `{"APPROVE", "REVIEW"}`. `REJECT` is an impossible role consequence for `compliance`.

### 2.2 When a Specialist Output Becomes `UNK` (`UNCERTAIN`)
A role's obligation status MUST become `UNK` (rule consequence forced to `REVIEW`)—and NEVER `SAT` (`APPROVE`) or substantive `FAIL` (`REJECT`)—whenever ANY of the following holds:
- Execution failure (`execution_error`, network/API timeout, exhausted bounded retries),
- Unparseable non-JSON text or schema violation (missing keys, non-finite/out-of-bounds `confidence`, unknown `decision` token not coerced),
- Impossible role consequence (`policy` returning `REVIEW`, or `evidence`/`risk`/`compliance` returning `REJECT`),
- Stale binding (`output.input_reference != authoritative binding` on `artifact_id`, `version`, or `content_hash`),
- Copied-fact mismatch (`output.evidence != authoritative source facts`),
- Rule citation mismatch (`output.rule_references != [expected_contract_version]`),
- Contradictory decision (`output.decision != expected_rule_decision(role, facts)`).

*Explicit Limit Note:* Free-text `reasoning_summary` non-emptiness is checked structurally, not semantically; a syntactically valid output with matching `decision`, `evidence`, and `rule_references` passes structural validation even if `reasoning_summary` is unfaithful (`F04b` transparency invariant).

---

## 3. Precedence Truth Table Mapping `(O_policy, O_evidence, O_risk, O_compliance) -> Final Decision`

Let `O = (O_p, O_e, O_r, O_c)` where each `O_x in {"SAT", "FAIL", "UNK"}`.

### 3.1 Default Mode: `substantive_exclusion`
Rules are evaluated strictly top-to-bottom:
1. **Policy Uncertainty Halt (`POLICY_UNCERTAIN`):** If `O_p == "UNK"` -> **`REVIEW`** (never guess `APPROVE` or `REJECT` without validated policy obligation).
2. **Authoritative Substantive Exclusion (`SUBSTANTIVE_EXCLUSION`):** Else if `O_p == "FAIL"` -> **`REJECT`** (confirmed policy exclusion overrides any `SAT`, `FAIL`, or `UNK` in `evidence`, `risk`, or `compliance`).
3. **Procedural Deferral (`PROCEDURAL_DEFERRAL`):** Else (`O_p == "SAT"`): if any of `{O_e, O_r, O_c}` is `"FAIL"` or `"UNK"` -> **`REVIEW`**.
4. **All Obligations Satisfied (`ALL_OBLIGATIONS_SATISFIED`):** Else (`O_p == O_e == O_r == O_c == "SAT"`) -> **`APPROVE`**.

### 3.2 Conservative Mode: `strict_audit`
1. **Any System Uncertainty Halt (`STRICT_AUDIT_UNCERTAINTY`):** If any `O_x == "UNK"` for `x in {policy, evidence, risk, compliance}` -> **`REVIEW`**.
2. **Substantive Exclusion (`SUBSTANTIVE_EXCLUSION`):** Else if `O_p == "FAIL"` -> **`REJECT`**.
3. **Procedural Deferral (`PROCEDURAL_DEFERRAL`):** Else if any of `{O_e, O_r, O_c}` equals `"FAIL"` -> **`REVIEW`**.
4. **All Obligations Satisfied (`ALL_OBLIGATIONS_SATISFIED`):** Else -> **`APPROVE`**.

---

## 4. Malformed Outputs, Stale Bindings, Conflicts & Update/Recovery Paths

1. **Malformed Outputs & Exhausted Crashes:** Treated strictly as `UNK` (`validation_status="FAILED"`), never coerced to `APPROVE` and never converted into substantive `REJECT`.
2. **Stale Bindings & Outdated Certificates:** `assess_specialist`, `verify_composition_independently`, and `CertificateSystem.commit_decision` refuse outputs or certificates bound to superseded dependency versions or mismatched `content_hash` values.
3. **Conflicting Findings:** Non-policy procedural issues cannot mask a confirmed `policy` exclusion under `substantive_exclusion`. Compromised specialists trying to force `APPROVE` over a true deficiency are mapped to `UNK`, blocking `APPROVE` in both precedence modes.
4. **Update & Selective Recovery Paths (`v1 -> v2`):**
   - **Narrow single-role edit:** Only the edited `facts:<role>@2`, its `<role>_agent@2`, `round@2`, `round-certificate@2`, and `final@2` recompute; unchanged specialist roles reuse `@1`.
   - **Broad multi-role edit:** Only the changed subset of specialist roles recomputes; unchanged roles reuse `@1`.
   - **No-change edit:** Zero agent re-executions (`executions_after == executions_before`).
   - **Same-label value edit (e.g., `amount_inr` changes within `coverage_limit_inr`):** Content hash changes, triggering `policy_agent` + `round` + `final` recovery and re-verification even when the final class (`APPROVE`) stays identical.

---

## 5. Pure Ground-Truth Derivation Rule (Clean Inputs)

For pure clean claim evaluation (zero faults/crashes/adversarial inputs, `O_x in {"SAT", "FAIL"}`), `substantive_exclusion` and `strict_audit` agree identically:
- **`REJECT`** iff `O_policy == "FAIL"`
- **`REVIEW`** iff `O_policy == "SAT"` and at least one of `{O_evidence, O_risk, O_compliance}` is `"FAIL"`
- **`APPROVE`** iff `O_policy == O_evidence == O_risk == O_compliance == "SAT"`
