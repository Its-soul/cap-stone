# Specialist-Role Composition Contract (`composition-v1`)

- **Status**: FROZEN SPECIFICATION
- **Contract Version**: `composition-v1`
- **Base Commit**: `1dc61fd81d94e352ae8ae06349acba32b0c0c0d1`
- **Annotation Status**: `SYNTHETIC_PROVISIONAL` (no human approval claimed)

---

## 1. Goal and Problem Statement

In the legacy prototype, local agent outputs were treated as interchangeable votes on whole-claim approval. When non-policy specialists (evidence, risk, compliance) observed that their specific checks passed, they voted `APPROVE`. On legitimate policy exclusion claims (e.g., cosmetic surgery or unfulfilled waiting period), this produced a 3-to-1 majority in favor of `APPROVE`, conflicting with the policy exclusion (`REJECT`) and forcing the mandatory gate to defer to `REVIEW`. This "role-fusion category error" induced **42 false reviews (additional deferrals)** across 45 rejection trials.

The goal of this composition contract is to eliminate this semantic error:
- Specialist roles are preserved: `policy`, `evidence`, `risk`, and `compliance`.
- Specialists evaluate **role-specific mandatory obligations**, not whole-claim approval.
- A specialist finding indicates whether its mandatory requirement is `SATISFIED`, `FAILED`, or `UNCERTAIN`.
- A composition truth table composes validated findings into a whole-claim decision.
- Confidence weighting is retained strictly as an **advisory diagnostic**, with zero authority to override a mandatory obligation.

---

## 2. Roles, Obligations, and Permitted Consequences

The four specialist roles evaluate distinct, non-fungible domain obligations under their respective rule contracts:

| Role | Contract Version | Obligation Name | Evaluated Predicates | Finding if Met | Finding if Unmet | Permitted Rule Consequence | Impossible Consequences |
|---|---|---|---|---|---|---|---|
| **`policy`** | `coverage-v1` | `policy_eligibility` | `hospitalization_covered`, `waiting_period_satisfied`, `amount_inr <= coverage_limit_inr` | `SATISFIED` | `FAILED` | `APPROVE` (met) / `REJECT` (unmet) | `REVIEW` is impossible as domain outcome |
| **`evidence`** | `documents-v1` | `evidence_integrity` | `documents_available`, `invoice_matches` | `SATISFIED` | `FAILED` | `APPROVE` (met) / `REVIEW` (unmet) | `REJECT` is impossible (missing docs != fraud/exclusion) |
| **`risk`** | `fraud-screen-v1` | `screening_integrity` | `not duplicate_claim`, `not provider_flagged` | `SATISFIED` | `FAILED` | `APPROVE` (met) / `REVIEW` (unmet) | `REJECT` is impossible (screening flag != fraud proof) |
| **`compliance`** | `identity-consent-v1` | `regulatory_controls` | `identity_verified`, `consent_available` | `SATISFIED` | `FAILED` | `APPROVE` (met) / `REVIEW` (unmet) | `REJECT` is impossible (missing control != claim rejection) |

### Rejection of Impossible Consequences
Any raw output that reports a consequence outside its permitted set (e.g., `risk` asserting `REJECT` or `policy` asserting `REVIEW`) is rejected by the validator as an `IMPOSSIBLE_CONSEQUENCE`, marked `INVALID`, and treated as `UNCERTAIN`.

---

## 3. Execution Failures vs. Domain Findings

Execution integrity is strictly separated from domain findings:
- **`VALIDATED`**: The specialist output passed structured schema parsing, exact version/hash binding checks, evidence matching, rule citation verification, and consequence validity.
- **`INVALID` / `FAILED` / `MISSING` / `STALE`**: A backend exception, malformed output, evidence mismatch, or stale input binding.
- **Explicit Uncertainty Rule**: Any specialist assessment that is not `VALIDATED` is marked `obligation_result = "UNCERTAIN"` and `rule_consequence = "REVIEW"`. Failed or missing assessments are **never silently treated as satisfied**.

---

## 4. Truth Table and Precedence for Mixed Findings

Let $S_P, S_E, S_R, S_C$ represent the validated assessments of `policy`, `evidence`, `risk`, and `compliance`.

### Truth Table (Comprehensive States)

| $S_P$ (Policy) | $S_E$ (Evidence) | $S_R$ (Risk) | $S_C$ (Compliance) | Composed Decision | Rationale |
|---|---|---|---|---|---|
| `SATISFIED` (`APPROVE`) | `SATISFIED` (`APPROVE`) | `SATISFIED` (`APPROVE`) | `SATISFIED` (`APPROVE`) | **`APPROVE`** | All four mandatory obligations independently satisfied. |
| `FAILED` (`REJECT`) | `SATISFIED` (`APPROVE`) | `SATISFIED` (`APPROVE`) | `SATISFIED` (`APPROVE`) | **`REJECT`** | Confirmed policy exclusion, waiting period unsatisfied, or limit exceeded. Non-policy obligations satisfied. |
| `SATISFIED` (`APPROVE`) | `FAILED` (`REVIEW`) | `SATISFIED` (`APPROVE`) | `SATISFIED` (`APPROVE`) | **`REVIEW`** | Policy eligible, but required documents missing or invoice mismatch. |
| `SATISFIED` (`APPROVE`) | `SATISFIED` (`APPROVE`) | `FAILED` (`REVIEW`) | `SATISFIED` (`APPROVE`) | **`REVIEW`** | Policy eligible, but duplicate claim or flagged provider detected. |
| `SATISFIED` (`APPROVE`) | `SATISFIED` (`APPROVE`) | `SATISFIED` (`APPROVE`) | `FAILED` (`REVIEW`) | **`REVIEW`** | Policy eligible, but patient identity unverified or consent missing. |
| `FAILED` (`REJECT`) | `FAILED` (`REVIEW`) | Any | Any | **`REJECT`** *(Substantive)* | Substantive policy exclusion takes precedence over procedural document review. |
| `FAILED` (`REJECT`) | Any | `FAILED` (`REVIEW`) | Any | **`REJECT`** *(Substantive)* | Substantive policy exclusion takes precedence over fraud screening flags. |
| `FAILED` (`REJECT`) | Any | Any | `FAILED` (`REVIEW`) | **`REJECT`** *(Substantive)* | Substantive policy exclusion takes precedence over compliance flags. |
| `UNCERTAIN` (`REVIEW`) | Any | Any | Any | **`REVIEW`** | Policy eligibility unknown; claim cannot be approved or rejected. |
| `SATISFIED` (`APPROVE`) | `UNCERTAIN` (`REVIEW`) | Any | Any | **`REVIEW`** | Unverified/failed assessment cannot be silently assumed satisfied. |
| `FAILED` (`REJECT`) | `UNCERTAIN` (`REVIEW`) | Any | Any | **`REJECT`** *(Substantive)* / **`REVIEW`** *(Strict Audit)* | See Precedence Specifications below. |

---

## 5. Precedence Specifications

### Policy A: `SUBSTANTIVE_EXCLUSION` (Default Engineering Standard)
1. **Rule 1 (Policy Uncertainty)**: If `policy` assessment is `UNCERTAIN` (failed, malformed, or stale), decision is **`REVIEW`**.
2. **Rule 2 (Substantive Exclusion)**: If `policy` assessment is `VALID` and `FAILED` (`REJECT`), decision is **`REJECT`**. Substantive ineligibility under `coverage-v1` cannot be converted to coverage by procedural status, nor does a crashed OCR/screening backend make an excluded claim payable. Unresolved procedural exceptions are logged in the composition record for audit traceability.
3. **Rule 3 (Procedural Deferral)**: If `policy` is `SATISFIED` but any non-policy specialist is `FAILED` or `UNCERTAIN`, decision is **`REVIEW`**.
4. **Rule 4 (Approval)**: If all four specialists are `VALID` and `SATISFIED`, decision is **`APPROVE`**.

### Policy B: `STRICT_AUDIT` (Alternative Strict Audit Mode)
1. If ANY specialist assessment is `UNCERTAIN` (system failure/malformed output), decision is **`REVIEW`** (audit integrity halts decision before domain rules).
2. Else if `policy` is `FAILED`: decision is **`REJECT`**.
3. Else if ANY non-policy specialist is `FAILED`: decision is **`REVIEW`**.
4. Else: decision is **`APPROVE`**.

---

## 6. Advisory-Only Confidence Containment

Confidence scores assigned to specialist findings (e.g. 0.90 for policy, 0.85 for evidence) are recorded and may be aggregated into an **advisory diagnostic consensus**.
- **Zero Decisional Authority**: The advisory consensus has zero weight in deciding the claim outcome.
- **No Override**: High confidence in an approving non-policy specialist cannot override a policy rejection or a missing document flag.
- **No Certificate Authority**: Confidence scores cannot issue certificates or authorize decisions.
