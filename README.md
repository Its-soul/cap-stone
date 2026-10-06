# Certificate Recovery

A Python prototype for keeping verified work consistent when its inputs change.
It records exact artifact versions and their dependencies, invalidates affected
certificates, and rebuilds and reverifies affected work while preserving unrelated
valid branches.

```text
Dependency change -> affected work becomes invalid -> rebuild -> verify -> new certificate
```

The in-memory core provides a dependency DAG, immutable JSON artifacts, registered
verifier/rebuilder functions, explicit certificate lifecycles, a mock decision
commit gate, and a SHA-256 audit chain. 

## Setup and Installation

Python 3.11+ and PyYAML are sufficient for the core. Gradio/Hugging Face clients 
and PyTorch/Transformers are optional.

Run from the repository root. On Windows, use `py -3` in place of `python` if needed.

```sh
python -m venv .venv
# Activate: Windows PowerShell .\.venv\Scripts\Activate.ps1
# Activate: macOS/Linux source .venv/bin/activate
python -m pip install -e .
python -m pip install -r requirements-test.txt
```

To run the remote client and integration test suites (which block network connections),
install the remote extra:
```sh
python -m pip install -e ".[remote]"
```

## Basic Execution and Testing

To inspect the CLI without submitting anything:
```sh
python run_baseline.py --help
```

To execute the test suite (requires `requirements-test.txt`):
```sh
python -m pytest tests -q
```

To run the preparation scripts verification:
```sh
python scripts/validate_preparation.py
```

## Optional Private-Checkpoint Testing (V2)

The V2 tests exercise input cleaning, batching, frozen-threshold reporting,
and artifact tampering. Model calls are mocked; no weights or downloads are needed.

The private-checkpoint test is opt-in and only replays two validation examples.
In an existing model environment, install dependencies without the remote extra:
```sh
python -m pip install -e ".[test]"
```
Set `CERT_RECOVERY_V2_RUN` to a completed V2 bundle, then run:
```sh
pytest tests/integration/test_v2_checkpoint.py -q
```
It reads existing artifacts without training or modifying results. Without the 
environment variable, it reports a skip.

## Advisory Role of the Model

**Semantic advice cannot override deterministic verification.** Advice binds exact
artifact IDs, versions, hashes and text. Stale advice is rejected; unavailable or
ambiguous advice is ignored. Entailment does not automatically establish a required
dependency, and confidence cannot issue certificates or authorize decisions.

The model acts in an advisory capacity, while the certificate engine holds full authority.

## Current Limitations

This is a trusted-process, in-memory prototype. 
- Source changes and dependency edges must be supplied by callers.
- Hashes detect certain edits, but do not authenticate a checkpoint controlled by an attacker. 
- There is no durable storage, distributed transaction, or real external action implementation.
- Remote loaded weights remain unverified and confidence is advisory and not assumed calibrated.
- Domain generalization and full performance claims are unverified at this stage.

## Documentation

- [Architecture and API](docs/ARCHITECTURE.md)
- [Data Derivation](docs/DATA.md)
- [Model Selection](docs/MODEL.md)
- [Dependency Targets](docs/DEPENDENCY_TARGET.md)
