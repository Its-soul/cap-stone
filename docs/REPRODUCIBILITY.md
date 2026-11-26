# Reproducibility and Execution Guide

## 1. Environment Separation

The core engine and offline tests require only Python 3.11+ and PyYAML. Heavy ML libraries (`torch`, `transformers`) have conflicting `huggingface-hub` constraints (`<1.0` for `transformers==4.57.1` vs. `>=2.0` for `gradio-client==2.7.2`) and should remain in separate virtual environments.

- **Core, Tests & Remote Client (`.venv`):**
  ```sh
  python -m venv .venv
  # Activate: .\.venv\Scripts\Activate.ps1 (Windows) or source .venv/bin/activate (Linux/macOS)
  python -m pip install -e ".[remote,test]"
  ```
- **Optional CPU/Lab ML Environments (`pyproject.toml` extras):**
  ```sh
  python -m pip install -e ".[laptop]"     # Local CPU ML / 1-epoch benchmark
  python -m pip install -e ".[nvidia]"     # Multi-GPU cluster / LoRA / Optuna
  python -m pip install -r requirements-colab.txt  # Pinned Colab smoke/pilot environment
  ```

## 2. Primary Demonstration

Launch the single browser demonstration (`demo/index.html`) backed by the real `SpecialistCompositionAdjudication` engine:

```sh
python scripts/run_claim_demo.py
```

Open `http://127.0.0.1:8000/`. For optional live V2 dependency scores in the browser panel, run the server from an environment containing `torch` and `transformers` with `CERT_RECOVERY_V2_RUN` set to a completed V2 run bundle (or a single `_private/runs/candidate_v2_*` directory). If no checkpoint is present, the panel reports availability cleanly while the adjudication workflow stays active.

## 3. Offline CLI Workflows & Verification

All commands below run offline without network access, paid compute, or V3 activation:

```sh
# Full offline test suite (unit, adjudication, core, remote mock, ml pipeline)
python -m pytest tests -q

# End-to-end deterministic adjudication, tamper detection, and recovery demo
python scripts/run_insurance_prototype.py

# 8-arm factorial ablation + deterministic oracle reference (writes to results/)
python scripts/run_insurance_ablation.py --suite extended --seeds 42 101 2024 --output results/ablation/ablation_check.json

# Specialist-role composition benchmark
python scripts/run_composition_evaluation.py --suite extended --output results/composition/comp_check.json

# Validate public configs, pins, documentation links, and notebook syntax
python scripts/validate_preparation.py
python scripts/validate_notebooks.py
```

## 4. Research Benchmark Utilities (`research_benchmark/` & `config/research_config.yaml`)

- **Configuration:** Shared hardware guardrails and group training presets live in [config/research_config.yaml](../config/research_config.yaml).
- **Generate 12-Model Reports:** Run `python scripts/generate_research_reports.py --target all --output-dir results` to export research matrices and evolution tables strictly into the ignored `results/` folder.
- **Canonical Public Notebooks (`notebooks/`):**
  1. `notebooks/01_Adjudication_Demo.ipynb` — Verified offline adjudication, fault containment, and recovery walkthrough.
  2. `notebooks/02_baseline_model.ipynb` — Disabled pinned NLI smoke template (all execution flags default to `False`).
  3. `notebooks/03_Dependency_Model_V2.ipynb` — Read-only Candidate V2 dependency model walkthrough.
  4. `notebooks/04_Model_Comparison_Laptop.ipynb` — 4-group laptop feasibility & historical benchmark outputs.
  5. `notebooks/05_Model_Comparison_NVIDIA.ipynb` — Multi-seed NVIDIA cluster benchmark scaffold (unexecuted).

## 5. Frozen Evidence & Safety Boundaries

- **V3 Remains Disabled:** `scripts/run_candidate_v3.py` requires external human approval and authorization files matching exact launch-spec hashes. Templates are rejected before execution.
- **No Unintended Overwrites:** Output helpers use exclusive file creation or write to `results/` (ignored by Git). Private checkpoints, run logs, and frozen datasets stay under `_private/` (ignored by Git) and are never modified by public test commands.
- **Optional Private V2 Checkpoint Replay:** In an ML environment with `pytest` installed:
  ```sh
  CERT_RECOVERY_V2_RUN=_private/runs/candidate_v2_<id> pytest tests/integration/test_v2_checkpoint.py -q
  ```
