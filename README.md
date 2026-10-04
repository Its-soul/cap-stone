# Dependency-aware certificate invalidation and recovery

Foundation prototype. The core works without an LLM, GPU, network, or database.
PAFA is reference material only; this implementation uses the scoped certificate design.

## Run code correctness tests

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/validate_preparation.py
```

These are small deterministic tests, not research experiments. No model packages
are needed. See `tests/test_engine.py` for a complete API usage example.
For a local editable install (including Windows), use `python -m pip install -e .`
and then `python -m unittest discover -s tests -v` without `PYTHONPATH`.

## Included

- `src/cert_recovery/`: deterministic core plus guarded future data/model/evaluation code.
- `config/`: centralized settings and ten experiment definitions.
- `notebooks/`: the eight requested unexecuted Colab notebooks.
- `tests/`: deterministic correctness tests, including guard and data-leakage checks.
- `docs/`: status, TODO, architecture, data/model plans, experiments and decisions.
- `data/examples/`: tiny authored input-format examples, not a research dataset.
- `results/`: execution-status manifest only; no experimental results.

## Future manual Colab workflow

Upload the project ZIP and the notebook you want to Colab. The first notebook
cell extracts the ZIP and locates the project. Read each notebook's setup notes.
All eight notebooks have empty outputs and no execution counts. Model/research
calls are guarded by flags that default to false.

**Prepared but intentionally not executed. Run manually in Google Colab.**
Model training, model evaluation, and research experiments are **NOT RUN**.

Start with `docs/PROJECT_STATUS.md`, `docs/ARCHITECTURE.md`, and `docs/TODO.md`.
Dataset annotations and external benchmark adapters remain future work.
Do not interpret configured cost units or unit-test assertions as research results.
Hashes detect edits relative to retained content/checkpoints; they do not authenticate
a maliciously rewritten checkpoint or establish semantic/causal truth.
