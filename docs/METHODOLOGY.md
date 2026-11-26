# Research Methodology and Dependency Formulation

## Binary Dependency Target (`depends_on` vs. `independent`)

For a candidate source artifact $S$ and named target artifact $T$:

- **Label `1` (`depends_on`):** $S$ is in $T$'s explicitly declared transitive support ancestry within the DAG.
- **Label `0` (`independent`):** An exhaustive support contract establishes that $S$ is absent from $T$'s declared ancestry. Incomplete scope is left unannotated rather than labeled negative.

| Question | Meaning | Defines Binary Dependency Target? |
|---|---|---|
| **Declared certificate support** | Membership in $T$'s declared transitive ancestry | **Yes** |
| **Value sensitivity** | Changing $S$ alters a computed output value | No |
| **Minimal necessary evidence** | Removing $S$ invalidates every proof | No |
| **Semantic NLI** | Premise entails, contradicts, or is neutral to hypothesis | No |
| **Lexical similarity** | Documents share overlapping vocabulary | No |

Reading a source with coefficient zero still binds it as declared support, and two OR witnesses remain declared supports even if one suffices. Conversely, quoting a fact from $S$ in background context does not establish computational support for target $T$.

## Evolution of the Custom Dependency Classifier (Stages A–F)

1. **Stage A — Off-the-Shelf NLI Baselines:** Pretrained NLI cross-encoders (`nli-MiniLM2-L6-H768`, `nli-deberta-v3-small`) predict semantic entailment $P(\text{Hypothesis} \mid \text{Premise})$. On dependency classification benchmarks, the pretrained NLI proxy failed to detect declared dependencies (50.0% accuracy, 0.333 macro-F1 on pilot test split).
2. **Stage B — Separating Entailment from Provenance:** Semantic entailment and topical overlap frequently occur between computationally independent documents, while genuine numerical or structural dependencies can hold with low surface word overlap. Because missing a true dependency risks reusing a stale certificate ($\text{FNR}_{\text{dep}}$ asymmetry), a dedicated binary provenance classifier is required.
3. **Stage C — Custom Candidate V2 Adaptation:** `cross-encoder/nli-deberta-v3-small` was adapted with a fresh seeded binary classification head (`independent`, `depends_on`) and trained for 2 epochs (256 optimizer steps, AdamW, `lr=2e-5`, batch size 2, max sequence length 128) on the frozen controlled dataset (`256 train / 64 val / 96 final + 384 stress`). Validation selected threshold $\tau = 0.20$.
4. **Stage D — Empirical Audit & Confound Discovery:** On held-out evaluation, V2 reached 87.5% accuracy (0.873 macro-F1) on validation, 91.7% accuracy (0.916 macro-F1, 0 false negatives) on the 96-row final split, and 82.8% accuracy (0.823 macro-F1, 42 improvements, 49 regressions) on the 384-row stress suite. However, auditing revealed a **critical confound**: V2 failed on all **16 `nli_context` independent pairs** where the hypothesis restated a factual claim from the candidate artifact despite zero declared arithmetic support, exploiting literal semantic truth as a shortcut.
5. **Stage E — Candidate V3 Root-Explicit Contrast Formulation (`PREPARED_NOT_RUN`):** To decouple semantic truth from DAG provenance, the Candidate V3 proposal (`src/cert_recovery/v3_preparation.py`, `src/cert_recovery/contrast_data.py`) adds an explicit scope boundary, separate observation check, and exhaustive negative closure clause (`"All sources have no supports."`) to both inputs across 512 total rows (`256 train / 64 val / 128 future-final` + 64 mentor contrast rows and 576 transformation links). Tokenizer audits confirm max 113/128 tokens with zero truncation. **Candidate V3 remains prepared and unexecuted**, gated behind explicit human review and authorization (`scripts/run_candidate_v3.py`).
6. **Stage F — Advisory Integration:** Deployed V2 predictions operate strictly as read-only advisory signals (`src/cert_recovery/v2_inference.py`); deterministic BFS traversal over SHA-256 DAG edges remains authoritative.

## Dataset Compatibility and Leakage Prevention

| Category | Dataset / Suite | Task Group | Leakage & Integrity Controls |
|---|---|---|---|
| **A. NLI & Semantic Advice** | `baseline_tests.json` (6 pairs), MultiNLI, ContractNLI | Group 1 | Diagnostic smoke pairs excluded from training |
| **B. Dependency Provenance** | Controlled Pilot (96), Candidate V2 (416 + 384 stress), Candidate V3 (512, unrun) | Group 2 | Family/world group isolation, normalized duplicate rejection, structural projection audit |
| **C. Evidence & Faults** | FEVER, HaluEval, synthetic corrupted agent outputs | Group 3 | Clean vs. corrupted schema perturbation isolation |
| **D. Generative Specialists** | InsuranceQA, CUAD, compliance calibration scenarios | Group 4 | Fixed prompt contract and schema validator (`ModelAgentAdapter`) |
| **F. Provenance & Recovery** | Deterministic Graph & Crypto Harness | Non-ML Core | Tested via exact BFS reachability and SHA-256 verification (no ML model used) |
