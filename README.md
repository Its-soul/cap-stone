# Certificate Recovery

## What is this project?

This final-year project explores how software agents can make decisions without
blindly trusting each other. An **agent** is a program with a specific job, such
as checking an insurance policy. We check their results, record the supporting
evidence, and rebuild only affected work after failures or changes. Today's demo
uses fixed-rule simulators; independent AI models are planned.

## The Problem

One wrong or compromised agent could mislead the others reviewing an insurance
claim, causing an unfair approval or rejection. We want to catch those mistakes,
explain decisions, and recover without repeating valid work.

## What We Are Building

```text
Insurance claim
      ↓
Policy, evidence, risk and compliance agents
      ↓
Check evidence, rules and confidence
      ↓
Flag faulty results and exclude invalid votes
      ↓
Combine valid votes and apply mandatory rules
      ↓
Record the decision and its supporting history
      ↓
If something fails or changes, rebuild only affected work
```

Confidence influences the proposed decision, but cannot override required rules.
When the proposal conflicts with those rules, the system returns **REVIEW**.

## Why This Is Different

- **Several roles:** separate checks for coverage, documents, risk and compliance.
- **Checked votes:** use confidence and validation, rather than just counting votes.
- **Deliberate failure testing:** simulate wrong, misleading and broken outputs.
- **Traceable history:** link decisions to evidence and detect edits to saved records.
- **Selective recovery:** reuse valid work instead of restarting everything.

## Current Implementation

- Four insurance agents returning structured decisions, confidence, evidence and rule references.
- Confidence-weighted voting and preliminary fault detection across six simulated failure modes.
- Linked records of rules, inputs, agent actions, votes and decisions, with tamper checks.
- Versioned verification records and repeatable recovery of affected work.
- A simple majority-vote baseline and offline experiments with configurable fault rates.

## Current Status

This is a research project still under development, not a production insurance service.

### Implemented

The features above run offline and have behavioral tests.

### Prototype / In Progress

- Agents are **deterministic simulators**: the same facts produce the same rule-based answer.
- Confidence values are manually assigned, not yet calibrated (checked against observed accuracy).
- Fault detection checks explicit rules; it does not establish malicious intent.
- Some coverage exclusions return REVIEW rather than an automatic rejection.
- Tamper checks need a trusted saved checkpoint; replacing both the history and checkpoint defeats them.

### Future Work

Formal Byzantine fault-tolerance guarantees (proofs about behavior despite faulty
or malicious agents) remain future work. **Robustness at 40% adversarial
interference is a research target, not a proven result.**

## Example

A synthetic **₹50,000** hospitalization claim has a **₹100,000** policy limit.
Coverage, waiting period, documents, risk and identity checks pass. The agents
recommend approval and the system records APPROVE.

If the risk agent fails, its output is excluded. Once the failure is cleared,
recovery reruns it and updates decision records, reusing the other three agents' work.

## Technical Overview

The core uses **Python 3.11+ and PyYAML**. JSON records hold exact artifact versions
and their dependencies. A certificate records that a registered verification rule
passed; final decisions require valid supporting certificates.

SHA-256 hashes provide digital fingerprints for the audit chain and **event DAG**,
a graph linking events to their contributors. These support tamper checks and
rule tracing, not complete causal inference.

Optional model advice uses exact input versions and hashes: stale advice is
rejected, and unavailable or ambiguous advice is ignored. Confidence cannot issue
certificates, authorize decisions or establish dependencies by itself.

The system runs in memory in a trusted process. Callers supply source changes and
dependencies. There is no durable storage, distributed transaction or real external
claim-processing action; remote model weights and broader performance remain unverified.

## Running the Project

Run from the repository root. On Windows, use `py -3` instead of `python` if needed.

```sh
python -m venv .venv
# Activate: Windows PowerShell .\.venv\Scripts\Activate.ps1
# Activate: macOS/Linux source .venv/bin/activate
python -m pip install -e .
python -m pip install -r requirements-test.txt
```

Run the insurance, failure, tamper-detection, recovery and comparison demos:

```sh
python scripts/run_insurance_prototype.py
```

Run the small synthetic experiment:

```sh
python scripts/run_insurance_prototype.py --experiments --trials 20 --seed 42
```

Add `--output path/to/new.json` to save records without overwriting files.
These offline commands neither enable V3 nor call remote models.

For the optional remote client and its offline test dependencies:

```sh
python -m pip install -e ".[remote]"
python run_baseline.py --help
```

The help command submits nothing. PyTorch/Transformers are optional and only
needed for model workflows; see [model setup](docs/MODEL.md).

## Tests

The [recorded milestone validation](docs/RESEARCH_PROTOTYPE.md#validation-record)
reports **319 passed, 1 skipped and 75 subtests passed**, including 48 insurance
tests. These are the previous implementation run's results; tests were not rerun
for this README edit.

```sh
python -m pytest tests -q
python -m pytest tests/test_insurance_adjudication.py -q
python scripts/validate_preparation.py
```

Most V2 tests use mocks, without model weights. The optional private-checkpoint
test replays two saved examples without training or changing results. In an
existing model environment:

```sh
python -m pip install -e ".[test]"
```

Set `CERT_RECOVERY_V2_RUN` to a completed V2 bundle, then run:

```sh
pytest tests/integration/test_v2_checkpoint.py -q
```

Without that environment variable, this test skips.

## Research Direction / Future Work

Next steps are stronger independent models, confidence checked against actual
accuracy, more realistic adversarial tests, stronger fault-tolerance evaluation,
authenticated decision histories and larger experiments.

The [original project description](docs/PROJECT_PROPOSAL.md) is preserved separately.
For further detail:

- [Prototype scope, limitations and measured results](docs/RESEARCH_PROTOTYPE.md)
- [Architecture and API](docs/ARCHITECTURE.md)
- [Data formats](docs/DATA.md)
- [Model setup and advice](docs/MODEL.md)
- [What counts as a dependency](docs/DEPENDENCY_TARGET.md)
