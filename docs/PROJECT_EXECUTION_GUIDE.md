# Certificate Recovery: Evidence-Based Project Guide

## Current verified state

This repository is a deterministic certificate-recovery prototype with an optional
semantic-dependency research pipeline. Its central guarantee is process-oriented:
when a declared dependency changes, affected certificates become invalid and a
registered recovery recipe must rebuild and reverify them before a decision can be
committed.

The following checks were executed locally without model loading, network access,
or writes to project data directories:

| Check | Verified result |
|---|---|
| Certificate recovery lifecycle | A changed source invalidated `cert@1`; recovery rebuilt `claim@2`, `cert@2`, and `decision@2`; the new certificate was valid. |
| Workflow policies | Independent update, cascade update, and untrusted-source cases were evaluated for all four policies. |
| Controlled candidate-v2 generator | 416 rows, 416 linked transformations, 256/64/96 train/validation/test split, balanced binary labels; leakage audit `PASS`. |
| V3 contrast generator | 64 rows and 72 transformations, balanced across all six NLI/dependency categories; audit `PASS`. |
| V3 candidate generator | 448 rows and 504 transformations, 256/64/128 train/validation/future-final split; audit `PASS`. |

These are software/data-generation checks, not evidence of trained-model accuracy.
No committed raw research dataset, trained checkpoint, or completed model-result
bundle is present in the repository.

## What the project measures

### Deterministic system experiment

The system compares four policies after a declared source changes:

| Policy | Interpretation |
|---|---|
| `A_full_recomputation` | Recreate the workflow from the changed inputs. |
| `B_no_invalidation` | Deliberately stale control; it can commit unsafe decisions. |
| `C_invalidation_only` | Invalidates affected work but does not rebuild it. |
| `proposed` | Plans selective recovery, rebuilds affected artifacts, re-verifies them, and commits only valid decisions. |

Report `safe_commits`, `total_commits`, `useful_completion_rate`, `blocked_tasks`,
`configured_work_units`, local elapsed time, declared impact precision/recall, and
recovery success rate. Work units are configured effort estimates, not money or
wall-clock time.

### Optional semantic-dependency experiment

The binary label is **not** ordinary NLI and is not causal necessity:

* `1` = the candidate source is in the named target artifact's explicitly declared
  transitive support ancestry.
* `0` = an exhaustive declared-support contract proves it is absent.
* incomplete context = unknown; do not silently label it `0`.

Model confidence remains advisory. It cannot issue certificates, override a
deterministic verifier, or authorize a decision.

## Required tables for a final report

| Table | Required columns |
|---|---|
| Dataset composition | dataset version, split, rows, groups/worlds, label-0, label-1, scenarios, template families |
| Data integrity | normalization rule, duplicates removed, group overlap, text/template overlap, transformations, dataset hashes |
| Main classifier results | method, split, threshold, accuracy, macro F1, precision, recall, F1, TP, FP, FN |
| Confusion matrix | method, split, actual class, predicted independent, predicted depends-on |
| Scenario/category results | scenario or NLI×dependency category, support, accuracy, macro F1, FN rate, FP rate |
| Transformation robustness | transformation kind, numerator, denominator, rate, both-correct rate, unique-row coverage |
| Workflow policy comparison | policy, cases, repetitions, safe/total commits, useful completion, blocked tasks, work units, elapsed time |
| Recovery correctness | impact precision, impact recall, invalidated certificate ratio, recovery success, rebuilt artifacts, audit events |
| Error analysis | pair ID, expected label, prediction, probability, threshold, scenario, error category, human-review outcome |
| Reproducibility/provenance | code hash, dataset hash, split hash, model/revision, tokenizer, seed, hardware, run status |

Never fill model-result tables with generator counts or notebook demonstrations.
Label all unavailable measurements as `NOT RUN`, with the reason.

## Ready-to-review input examples

Save reviewed rows as `data/raw/dependency_pairs.jsonl` only after the labels and
scope have been checked by a domain reviewer:

```json
{"pair_id":"dep-001","world_id":"plant-01","premise":"For target Q only, Q declares Oak and Pine as its exhaustive direct supports. All sources have no supports. Oak reads 12. Pine reads 8.","hypothesis":"For target Q only, Oak is a declared transitive support.","label":1,"source":"reviewed-authored-example"}
{"pair_id":"dep-002","world_id":"plant-01","premise":"For target Q only, Q declares Oak and Pine as its exhaustive direct supports. All sources have no supports. Oak reads 12. Pine reads 8.","hypothesis":"For target Q only, Elm is a declared transitive support.","label":0,"source":"reviewed-authored-example"}
```

Use separate groups/worlds for train, validation, and test. Keep every paraphrase,
renaming, or support-substitution counterpart in the same split group.

For the deterministic workflow, save cases as `data/raw/workflow_cases.jsonl`:

```json
{"world_id":"case-independent-update","topology":"independent","sources":[{"value":4},{"value":9}],"change":{"source_index":0,"kind":"update","value":7}}
{"world_id":"case-cascade-update","topology":"cascade","sources":[{"value":3},{"value":5},{"value":2}],"change":{"source_index":0,"kind":"update","value":6}}
{"world_id":"case-untrusted-source","topology":"independent","sources":[{"value":4},{"value":9}],"change":{"source_index":0,"kind":"untrusted","value":4}}
```

## Safe execution sequence

1. Create a Python 3.11+ virtual environment and install the project plus test
   dependencies.
2. Run `python -m pytest tests -q` and keep the output with the report.
3. Run `python scripts/validate_preparation.py`.
4. Review and supply raw inputs. Do not use the examples as research evidence.
5. Run deterministic workflow experiments only after explicitly enabling the
   corresponding configuration flag.
6. Run a model experiment only with an immutable model revision, a fresh empty
   output directory, reviewed data, and a frozen validation selection before test
   inference.
7. For V3, complete the required external human-review and authorization sidecars
   before invoking `scripts/run_candidate_v3.py`.

## Known boundaries

* The core is in-memory; it is not durable or distributed.
* Dependencies are caller-declared; the system does not discover missing edges.
* Hashes detect tampering but do not authenticate an attacker-controlled store.
* Synthetic labels and small controlled data do not establish real-world causal
  dependency, calibration, or broad generalization.
* Remote and direct model results are distinct and must never be combined.
