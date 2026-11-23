# Specialist-Role Composition Evaluation Plan

- **Status**: FROZEN SPECIFICATION (Before Execution)
- **Contract Version**: `composition-v1`
- **Annotation Status**: `SYNTHETIC_PROVISIONAL` (no human approval claimed)
- **Base Commit**: `1dc61fd81d94e352ae8ae06349acba32b0c0c0d1`

---

## 1. Objective and Comparison Arms

This evaluation compares the repaired experimental baseline with the newly implemented specialist-role composition prototype to verify whether the semantic "role-fusion category error" (which induced 42 false reviews across 45 rejection presentations in legacy Arm E) is eliminated, while strictly preserving deterministic safety and selective recovery.

### Compared Methods:
1. **Method E (Legacy Full Prototype)**: Preserved legacy implementation utilizing confidence weighting + output validation + mandatory gate (`InsuranceAdjudication(claim, mode="legacy")`). Evaluates whole-claim voting aggregation.
2. **Method COMP (Specialist-Role Composition)**: New prototype (`SpecialistCompositionAdjudication(claim, mode="composition", precedence="substantive_exclusion")`). Evaluates typed specialist obligations composed via the frozen `composition-v1` truth table. Advisory confidence is retained as a diagnostic comparator with zero decisional authority.
3. **Method F (Deterministic Rules Only)**: Ground-truth domain rules evaluated directly on claim facts. Serves as the pure oracle baseline.

---

## 2. Evaluation Suites

Three bounded, frozen fixture suites are evaluated:

### Suite 1: Extended Scenario Suite (`extended`)
- **Trial Count**: 150 trials (10 scenarios × 15 presentations).
- **Random Seed**: 42 (identical to repaired baseline).
- **Corruption Schedule**:
  - Run 0: Clean (no corruption).
  - Runs 1–14: Controlled fault injection across 4 corruption levels (0%, 25%, 50%, 75%) sampled across `FAULT_MODES`.
- **Scenarios**:
  - `ext_0`: Standard clean claim (`APPROVE`)
  - `ext_1`: High-value clean claim within limit (`APPROVE`)
  - `ext_2`: Hospitalization not covered (`REJECT`)
  - `ext_3`: Waiting period unsatisfied (`REJECT`)
  - `ext_4`: Claim amount exceeds policy limit (`REJECT`)
  - `ext_5`: Required documents unavailable (`REVIEW`)
  - `ext_6`: Invoice does not match records (`REVIEW`)
  - `ext_7`: Duplicate claim flagged (`REVIEW`)
  - `ext_8`: Healthcare provider flagged (`REVIEW`)
  - `ext_9`: Patient consent unavailable (`REVIEW`)

### Suite 2: Historical Prototype Suite (`historical`)
- **Trial Count**: 75 trials (5 scenarios × 15 presentations).
- **Random Seed**: 42.
- **Scenarios**:
  - `hist_0`: Clean baseline (`APPROVE`)
  - `hist_1`: Policy exclusion (`REJECT`)
  - `hist_2`: Missing documents (`REVIEW`)
  - `hist_3`: Duplicate claim (`REVIEW`)
  - `hist_4`: Missing identity (`REVIEW`)

### Suite 3: Mixed-Condition & Uncertainty Suite (`mixed`)
- **Trial Count**: 60 trials (6 scenarios × 10 presentations).
- **Random Seed**: 101.
- **Scenarios**:
  - `mix_0_exclusion_and_missing_docs`: Hospitalization not covered + missing documents (Label: `REJECT` under substantive exclusion).
  - `mix_1_exclusion_and_duplicate_claim`: Waiting period unsatisfied + duplicate claim flagged (Label: `REJECT`).
  - `mix_2_exclusion_and_missing_consent`: Amount exceeds limit + missing consent (Label: `REJECT`).
  - `mix_3_clean_policy_missing_docs_and_risk`: Clean policy facts + missing documents + provider flagged (Label: `REVIEW`).
  - `mix_4_policy_exclusion_and_faulty_evidence`: Hospitalization not covered + evidence agent execution failure (Label: `REJECT` under substantive exclusion; `REVIEW` under strict audit).
  - `mix_5_clean_policy_and_faulty_compliance`: Clean policy facts + compliance agent execution failure (Label: `REVIEW`).

---

## 3. Metrics Specification

1. **Three-Class Accuracy**: $\frac{\text{Correct Predictions}}{\text{Total Presentations}}$
2. **Decisive Coverage**: $\frac{\text{Decisive Outcomes (APPROVE or REJECT)}}{\text{Total Presentations}}$
3. **Decisive Accuracy**: $\frac{\text{Correct Decisive Predictions}}{\text{Decisive Outcomes}}$ (Undefined if Decisive Outcomes = 0)
4. **Required Reviews**: Predictions of `REVIEW` on cases where ground-truth is `REVIEW`.
5. **Additional Deferrals (False Reviews)**: Predictions of `REVIEW` on cases where ground-truth is decisive (`APPROVE` or `REJECT`).
6. **Unsafe Approvals**: Predictions of `APPROVE` on cases where ground-truth is `REJECT` or `REVIEW`.
7. **Incorrect Rejections**: Predictions of `REJECT` on cases where ground-truth is `APPROVE` or `REVIEW`.

---

## 4. Preservation and Overwrite Guards

- All historical experiment bundles in `_private/runs/` and `results/ablation/` are frozen and read-only.
- New evaluation outputs must be saved exclusively to `results/composition/`.
- Overwrite guards verify target files do not overwrite previous historical benchmarks.
