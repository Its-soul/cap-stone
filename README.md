# Certificate recovery

A Python prototype for keeping verified work consistent when its inputs change.
It records exact artifact versions and their dependencies, invalidates affected
certificates, and rebuilds and reverifies affected work while preserving unrelated
valid branches.

```text
Dependency change -> affected work becomes invalid -> rebuild -> verify -> new certificate
```

The in-memory core provides a dependency DAG, immutable JSON artifacts, registered
verifier/rebuilder functions, explicit certificate lifecycles, a mock decision
commit gate, and a SHA-256 audit chain. Python 3.11+ and PyYAML are sufficient for
the core; Gradio/Hugging Face clients and PyTorch/Transformers are optional.

## Install and test

Run from the repository root. On Windows, use `py -3` in place of `python` if needed.

```sh
python -m venv .venv
# Activate: Windows PowerShell .\.venv\Scripts\Activate.ps1
# Activate: macOS/Linux source .venv/bin/activate
python -m pip install -e ".[remote]"
python -m unittest discover -s tests -q
python -m unittest discover -s tests/remote -q
python -m unittest discover -s tests/integration -q
python scripts/validate_preparation.py
```

`python -m pip install -e .` installs only the core. The remote extra is needed
for the client and integration test suites; those tests block network connections.
The current suites contain **56 core/preparation, 40 client, and 14 integration
tests (110 unique tests)**. They establish behavior on fixtures, not model quality
or research performance. See [architecture and API](docs/ARCHITECTURE.md) for
using the engine and its authority boundary.

## Configuration and optional models

[`config/config.yaml`](config/config.yaml) defines paths, recovery costs/budgets,
and optional data/model settings. All workload flags default to false. Model,
data-preparation and research entry points also require `manual=True`; ordinary
core operations do not.

The optional client sends premise/hypothesis pairs to reviewed Hugging Face
Spaces, validates their API/response contracts, and writes a fresh timestamped
JSON result for every run. It downloads no weights. Copy `.env.example` to `.env`
only when needed; anonymous access is supported, or set a Hugging Face READ token.
Never commit credentials. Inspect the CLI without submitting anything:

```sh
python run_baseline.py --help
```

Future remote calls require explicit alternative-model opt-in, available free
service/quota, and authorization for inference. Setup, response semantics and
pinned direct-model preparation are described in [model/client guidance](docs/MODEL.md).
The six fixed cases and minimal recorded-response fixture are public; generated
logs and private datasets are excluded. Data formats are in [DATA.md](docs/DATA.md).

**Semantic advice cannot override deterministic verification.** Advice binds exact
artifact IDs, versions, hashes and text. Stale advice is rejected; unavailable or
ambiguous advice is ignored. Entailment does not automatically establish a required
dependency, and confidence cannot issue certificates or authorize decisions.

## Limits

This is a trusted-process, in-memory prototype. Source changes and dependency
edges must be supplied by callers. Verifiers/rebuilders must implement real task
rules. Hashes detect certain edits; they do not prove truth or authenticate a
checkpoint controlled by the same attacker. There is no durable storage,
distributed transaction or real external action implementation.

Remote loaded weights remain unverified. A retained fixture records contradiction
at 99.57% for a neutral pizza/bicycle pair; successful transport is not semantic
correctness. Confidence is advisory and not assumed calibrated. The optional
notebook is disabled and has no execution outputs. Colab compatibility and domain
generalization remain unverified. Direct model and synthetic pilot execution
statuses belong to each private run manifest, separately from notebook preparation
and historical Space logs.

## Optional bounded synthetic pilot

Use a separate environment for `requirements-colab.txt`; its Transformers/Hub
dependencies differ from the remote client extra. This deliberately executed
command downloads pinned weights, runs direct smoke comparisons, fine-tunes an
initial dependency head, and evaluates deterministic toy workflows:

```sh
python scripts/run_pilot.py --execute
```

Prefer a free Colab GPU using the disabled pilot section of the existing
[`02_baseline_model.ipynb`](notebooks/02_baseline_model.ipynb). CPU is supported.
The command creates a new ignored `_private/runs/pilot_<timestamp>_<id>/` with
statuses, inputs, family-isolated splits, model outputs/checkpoints, measured
metrics, CSVs, plots and a standalone `report.html`. Open that file directly;
no server is required. Every report is marked **PRELIMINARY / EXPERIMENTAL
— not final or production results.** Failed/unrun stages are explicit.

Rebuild the viewer without inference using
`python scripts/run_pilot.py --report-only --run <run-directory>` (with matplotlib
installed). Validate saved artifacts offline using
`python scripts/validate_pilot.py <run-directory>`; this blocks network connections.

The bounded preset uses 96 synthetic source-sum pairs (64/16/16 rows, seed 42,
12 template families), two training epochs, and 12 workflow cases repeated three
times for each existing policy. Smoke cases never enter training. Model stages
have a 30-minute limit and no automatic inference retry. Predictions remain
advisory. See [data derivation](docs/DATA.md) and [model selection](docs/MODEL.md).
