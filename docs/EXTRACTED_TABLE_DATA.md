# Extracted Table Data

This file is a presentation-ready inventory of the data that is currently
available from the repository and locally executed offline checks. It must not be
presented as a real-world deployment or trained-model evaluation.

## Table 1 — Current data and result availability

| Item | Rows / records | Status | Use in final report |
|---|---:|---|---|
| Authored NLI smoke cases | 6 | Present; fixed examples | Smoke/illustration only; not model training data |
| Dependency-pair format examples | 6 | Present | Input-schema illustration only |
| Workflow-case format examples | 3 | Present | Input-schema illustration only |
| Controlled V2 synthetic dependency data | 416 | Generated locally; leakage audit passed | Method/data-composition table only unless a separately authorized model run is completed |
| Controlled V2 transformation links | 416 | Generated locally | Robustness-protocol table |
| V3 contrast data | 64 | Generated locally; audit passed | NLI-vs-dependency design table; pending human review |
| V3 contrast transformations | 72 | Generated locally | Transformation protocol table |
| V3 candidate data | 448 | Generated locally; audit passed | Proposed-data table; pending human review |
| V3 candidate transformations | 504 | Generated locally | Proposed robustness table |
| Locally extracted dependency-pair seed data | 14 | Present in `data/raw/local_project_dependency_pairs.jsonl`; derived from this repository's `pyproject.toml` | Pipeline-input demonstration only; too small and single-project to be a benchmark |
| Public multi-repository dependency-pair seed data | 28 | Present in `data/raw/public_python_dependency_pairs.jsonl`; extracted from immutable public manifests for Requests, Flask, and HTTPX | Small direct-dependency pilot; not a generalizable benchmark |
| External real-world dependency data | 0 committed | Missing | Must be collected/reviewed before a generalizable real-data experiment |
| Trained checkpoint/results bundle | 0 committed | Missing | Do not create accuracy/F1 result claims |

## Table 1a — Raw dependency-pair pilot provenance

| Dataset component | Rows | Groups | Label 1 | Label 0 | Evidence source |
|---|---:|---:|---:|---:|---|
| Local project extraction | 14 | 3 | 7 | 7 | This repository's `pyproject.toml` |
| Requests extraction | 8 | 1 | 4 | 4 | Immutable `pyproject.toml`, commit `611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60` |
| Flask extraction | 10 | 1 | 6 | 4 | Immutable `pyproject.toml`, commit `d086db856be187255b8ec61ef409357393020f32` |
| HTTPX extraction | 10 | 1 | 4 | 6 | Immutable `pyproject.toml`, commit `b5addb64f0161ff6bfe94c124ef76f6a1fba5254` |
| Combined pilot | 42 | 6 | 21 | 21 | `data/raw/local_project_dependency_pairs.jsonl` + `data/raw/public_python_dependency_pairs.jsonl` |

The combined pilot passed the repository cleaner with 42 clean rows and no
duplicates. Grouped splitting at seed 42 produces 24 train, 10 validation, and 8
test rows; both binary classes appear in every split. This is enough to validate
input handling, but not enough to train or compare a meaningful classifier.

## Table 2 — Controlled V2 data composition

| Split | Rows | Label 1: depends-on | Label 0: independent | Template families |
|---|---:|---:|---:|---:|
| Train | 256 | 128 | 128 | 8 |
| Validation | 64 | 32 | 32 | 4 |
| Test | 96 | 48 | 48 | 6 |
| Total | 416 | 208 | 208 | 18 |

| Scenario | Rows |
|---|---:|
| Sum | 96 |
| Approval-all | 96 |
| Authentication-all | 96 |
| NLI-context | 96 |
| Weighted sum | 32 |

## Table 3 — V3 contrast matrix

| NLI relation | Dependency = 1 | Dependency = 0 | Total |
|---|---:|---:|---:|
| Entailment | 11 | 11 | 22 |
| Contradiction | 11 | 11 | 22 |
| Neutral | 10 | 10 | 20 |
| Total | 32 | 32 | 64 |

This table is essential: NLI entailment, contradiction, and neutrality do not
define declared dependency support.

## Table 4 — V3 proposed candidate composition

| Split | Rows | Purpose |
|---|---:|---|
| Train | 256 | Model fitting |
| Validation | 64 | Checkpoint and threshold selection only |
| Future-final | 128 | One-time post-selection comparison only |
| Total | 448 | Proposed, not executed |

| NLI × dependency category | Rows |
|---|---:|
| Entailment × depends-on | 75 |
| Entailment × independent | 75 |
| Contradiction × depends-on | 75 |
| Contradiction × independent | 75 |
| Neutral × depends-on | 74 |
| Neutral × independent | 74 |

## Table 5 — Executed deterministic workflow results

The exact row-level source is [workflow_policy_results.example.csv](../data/examples/workflow_policy_results.example.csv).

| Case | Policy | Safe / total commits | Useful completion | Work units | Interpretation |
|---|---|---:|---:|---:|---|
| Independent update | Full recomputation | 2 / 2 | 1.00 | 12 | Safe reference policy |
| Independent update | No invalidation | 1 / 2 | 0.50 | 0 | Cheap but commits stale work |
| Independent update | Invalidation only | 1 / 1 | 0.50 | 1 | Safe but blocks affected work |
| Independent update | Proposed | 2 / 2 | 1.00 | 7 | Selective recovery restored affected work |
| Cascade update | Full recomputation | 3 / 3 | 1.00 | 18 | Safe reference policy |
| Cascade update | No invalidation | 0 / 3 | 0.00 | 0 | All committed decisions were unsafe |
| Cascade update | Invalidation only | 0 / 0 | 0.00 | 1 | Safe blocking of affected cascade |
| Cascade update | Proposed | 3 / 3 | 1.00 | 17 | Full safe recovery with less configured work |
| Untrusted source | Proposed | 1 / 1 | 0.50 | 1 | Correctly blocks the branch that relies on untrusted input |

## Tables that must remain `NOT RUN` for now

Do not populate these until a model is actually run against reviewed data:

| Table | Current status |
|---|---|
| Fine-tuned model accuracy / precision / recall / macro F1 | NOT RUN |
| Model confusion matrix | NOT RUN |
| Probability calibration: Brier/ECE | NOT RUN |
| Model error-analysis rows | NOT RUN |
| Direct pinned-model smoke predictions | NOT RUN |
| Remote-model baseline results | NOT RUN |

## Raw-data fields to collect

For every real source graph or repository, preserve:

| Field | Why it is needed |
|---|---|
| `pair_id`, `world_id` | Row identity and split grouping |
| `premise`, `hypothesis` | Exact model-visible input |
| `label` | Declared-support target label only |
| `source` | Repository/provenance and licensing traceability |
| repository URL and immutable commit/version | Reproducibility |
| manifest/graph hash | Evidence integrity |
| target artifact and exhaustive support list | Justifies both positive and negative labels |
| candidate source | The dependency being judged |
| reviewer decision/rationale | Required for credible real-data annotation |
