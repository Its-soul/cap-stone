# Verifiable Multi-Agent Adjudication with Cryptographic Provenance

**`certificate-recovery`** is a Python verification kernel for fault-tolerant multi-agent decision workflows. It combines deterministic rule gating, independent verification against source data, Ed25519-signed causal audit logs, and dependency-aware selective recovery so updated inputs recompute only affected steps.

---

## Why This Exists

Multi-agent LLM and ML pipelines commonly fail in high-stakes workflows for three reasons:

- **Stale decisions:** When a single input record changes later, cached approvals or downstream summaries from an earlier version can be replayed against outdated facts.
- **Unverified agent outputs:** Free-form model outputs, crashed calls, or majority voting across unrelated checks can let one error or faulty agent approve a request that violates a core rule.
- **Costly full restarts:** Fixing one changed field normally forces the entire pipeline to re-run from scratch, wasting API calls and compute.

This project draws a strict boundary between **untrusted model proposals** and **authoritative host verification**. AI/ML backends only propose structured role assessments, while deterministic host code validates schemas, enforces domain rules, signs provenance records, and manages incremental updates.

---

## Key Features

- **Dependency-aware certificate invalidation and targeted recovery:** `CertificateSystem` tracks exact `(artifact_id, version, content_hash)` bindings. Editing a single fact invalidates only dependent branches while keeping unaffected specialist certificates valid.
- **Independent verification & fault-tolerant adjudication:** Four role-isolated specialists (`policy`, `evidence`, `risk`, `compliance`) evaluate separate obligations, and `independent_verifier` re-derives every decision directly from stored source ancestry before commit. Malformed, timed-out, stale, or rule-conflicting outputs fall back safely to `REVIEW` instead of `APPROVE`.
- **Ed25519 signed causal provenance & tamper detection:** Every input mutation, specialist output, recovery event, certificate state transition, and final decision is stored in a SHA-256 content-addressed `EventDAG` signed with Ed25519 keys alongside a `checkpoint.json` anchor.
- **Benchmark evaluation suite & offline reproducibility:** Includes the curated synthetic benchmark under [datasets/benchmark_v1/](datasets/benchmark_v1/), comparative baselines (`FULL_DET`, voting, policy-veto, single-agent, and LLM-as-a-judge harnesses), and `run_all.py` for deterministic offline regeneration.

---

## Repository Structure

```text
.
|-- config/                         # Model pins and evaluation settings
|-- data/examples/                  # Sample JSONL schemas for dependency pairs and workflows
|-- datasets/
|   |-- benchmark_v1/               # Synthetic claim splits, update paths, and SHA-256 manifest
|   `-- blind_external_template.json
|-- docs/                           # Architecture, decision spec, evaluation protocol, and model notes
|-- results/                        # Local evaluation and benchmark outputs (gitignored working dir)
|-- scripts/                        # CLI scripts for tamper tests, replay tests, audits, and benchmarks
|-- src/cert_recovery/              # Core DAG certificate engine, adjudication workflow, and oracle modules
|-- tests/                          # Unit, adjudication, core, ML pipeline, and remote contract suites
|-- pyproject.toml                  # Package metadata and optional dependency extras
`-- run_all.py                      # Offline reproducibility entry point
```

---

## Quick Start & Setup

Requires **Python 3.11+**. The core engine requires only `PyYAML` and `cryptography`.

### 1. Clone and install

```sh
git clone https://github.com/Its-soul/cap-stone.git
cd cap-stone
python -m venv .venv
```

Activate the virtual environment and install the package with test dependencies:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

```sh
# macOS / Linux
source .venv/bin/activate
python -m pip install -e ".[test]"
```

Optional extras (`remote`, `laptop`, `nvidia`) are defined in [pyproject.toml](pyproject.toml) if you run optional Hugging Face Space or local ML model scripts.

### 2. Run the full offline test suite

```sh
pytest tests -q
```

### 3. Run offline reproducibility & verification scripts

```sh
# Regenerate seeded adjudication tables and verify saved benchmark artifacts offline
python run_all.py --output-dir results/verify_tmp

# Verify stale-certificate replay rejection across v1 -> v2 claim updates
python scripts/run_stale_certificate_experiment.py --output-dir results/replay_tmp

# Generate signed EventDAG logs, run tamper/rollback tests, and inspect a persisted run
python scripts/run_provenance_tamper_experiment.py --output-dir results/provenance_tmp
python scripts/verify_provenance.py \
  --events results/provenance_tmp/persisted_run/events.jsonl \
  --public-key results/provenance_tmp/persisted_run/public_key.hex \
  --checkpoint results/provenance_tmp/persisted_run/checkpoint.json \
  --query-artifact facts:evidence@1

# Run the full offline synthetic benchmark suite against the hash-verified manifest
python scripts/run_benchmark_evaluation.py \
  --manifest datasets/benchmark_v1/frozen_evaluation_manifest.json \
  --results-dir results/benchmark_tmp \
  --run-offline
```

For opt-in live model runs (for example via Groq), copy `.env.example` to `.env`, set `GROQ_API_KEY`, and pass `--run-live` to [scripts/run_benchmark_evaluation.py](scripts/run_benchmark_evaluation.py) or configure [scripts/run_pilot_workflow.py](scripts/run_pilot_workflow.py).

---

## How It Works

```text
  Claim Input (10 typed fields across 4 role slices)
         |
         +--> [policy / evidence / risk / compliance]  (untrusted specialist proposers)
         |
         v
  Schema & Source-Binding Validation
         |   (missing / malformed / stale / conflicting output -> UNCERTAIN -> REVIEW)
         v
  Mandatory Rule Composition & Independent Verifier
         |   (policy failed -> REJECT | procedural gap -> REVIEW | all valid -> APPROVE)
         v
  Ed25519 Signed EventDAG + Checkpoint Anchor
         |   (tamper-evident causal log + certificate issued on verified bindings)
         v
  Source Fact Update (v1 -> v2)
         |   (invalidate affected branch only -> re-run dirty specialist -> reuse valid v1 certs)
         v
  Updated Certified Decision
```

1. **Input slicing:** A claim is split into disjoint role inputs so each specialist sees only the fields for its contract (`coverage-v1`, `documents-v1`, `fraud-screen-v1`, `identity-consent-v1`).
2. **Adjudication & validation:** Specialists return structured JSON proposals. Host code checks schemas, copied evidence fields, rule citations, and version hashes, mapping any fault to `UNCERTAIN`.
3. **Independent verification:** Before committing a decision, `independent_verifier` re-computes domain obligations directly from source artifacts in the DAG.
4. **Signed provenance:** Every step is appended to an Ed25519-signed `EventDAG` log paired with an external `checkpoint.json` to detect tampering, reordering, cross-run splicing, and history truncation.
5. **Selective recovery:** When a claim field updates from `v1` to `v2`, `CertificateSystem` invalidates downstream nodes on that branch, re-runs only invalidated specialists, reuses untouched `v1` outputs, and issues an updated certificate.

---

## Limitations & Scope

- **Synthetic benchmark domain:** All claim schemas and domain predicates represent a synthetic healthcare adjudication specification rather than live production insurance or clinical data.
- **Offline defaults vs. opt-in live LLM runs:** The test suite and offline scripts use deterministic rule/contract backends so everything runs with zero network access or API cost. Live LLM evaluation requires explicit opt-in flags and an API key.
- **CPU-only local execution:** Core tests, cryptographic provenance checks, and benchmark harnesses run locally on CPU without requiring a GPU.
- **Single-writer provenance model:** Ed25519 signatures verify possession of the local writer's signing key, and detecting tail truncation requires supplying an independently retained `checkpoint.json` file (this is not a distributed Byzantine consensus system).
- **Structural vs. free-text checks:** Host validation enforces exact structured decisions, evidence copies, rule references, and input hashes; verifying natural-language faithfulness inside free-text `reasoning_summary` strings remains outside automated checks.

---

## Documentation & Project Info

- [Architecture & Data Contracts](docs/ARCHITECTURE.md)
- [Frozen Decision Specification](docs/FROZEN_DECISION_SPEC_V1.md)
- [Evaluation Protocol](docs/FROZEN_EVALUATION_PROTOCOL_V1.md)
- [Evaluation Summary](docs/EVALUATION.md)
- [Methodology](docs/METHODOLOGY.md)
- [Models & Baselines](docs/MODELS.md)
- [Reproducibility Guide](docs/REPRODUCIBILITY.md)

**Package:** `certificate-recovery` (`0.1.0`) — see [pyproject.toml](pyproject.toml) for dependencies (`PyYAML`, `cryptography`, Python `>=3.11`). **Author:** Vaibhav Yadav. No separate repository license file is currently included.
