# Experiments — all NOT RUN

**Prepared but intentionally not executed. Run manually in Google Colab.**
Results for every experiment: **none**. Code correctness tests are separate.
Machine-readable objective/input/variables/baseline/method/metrics/expected-output
definitions are in `config/experiments.yaml`. No expected numerical performance
is asserted. Model training, model evaluation and research experiments were not executed.

| ID / goal | Baseline / method | Metrics | Notebook |
|---|---|---|---|
| E01 Change detection | Full restart; externally announced version event → impact report | Events, versions, affected count | 05 |
| E02 Direct invalidation | No invalidation; reviewed direct graph bindings | Set precision/recall, unnecessary invalidations | 06 |
| E03 Transitive invalidation | No invalidation; closure vs independent topology expectation | Impact precision/recall, invalidated ratio | 06 |
| E04 Selective recovery | Invalidation only; topological rebuild and fresh verification | Recovery rate, support-valid commits, useful completion | 07 |
| E05 Cost comparison | Full recomputation; fresh paired worlds and same obligations | Work units, latency, savings, useful completion | 08 |
| E06 Integrity | Unanchored hash chain; edit/reorder/truncate exported copies | Detection by attack, verification latency, provenance bytes | 06 |
| E07 Source replacement | Restart; block during failure, recover after trusted replacement | Blocked decisions, recovery rate, attempted cost | 07 |
| E08 Semantic advice | NLI proxy/declared edges; reviewed world-disjoint annotations | Accuracy, precision, recall, F1, Brier, ECE, false negatives | 04 |
| E09 Fine-tuned comparison | Pretrained proxy; validation selection then frozen test | Classification and calibration, error examples | 04 |
| E10 Fault injection | A/B/C/proposed; unavailable source, stale plan, missing recipe, budget exhaustion | Support-validity, utility, blocking, integrity | 08 |

Notebooks 01–03 prepare data, pretrained validation baseline and fine-tuning.
All eight notebooks have null execution counts and empty outputs.

## Baselines and adapter limits
- A: rebuild all work from current sources; no impact-based reuse.
- B: keep cached certificates while sources change outside that cache; deliberately
  unsafe control. Independent source/version oracle exposes stale commits.
- C: invalidate affected work and abstain; retain unrelated valid work.
- Proposed: invalidate, rebuild affected artifacts, verify, issue new certificates.

The arithmetic adapter uses explicit independent/cascade source-sum contracts.
It is runnable preparation, not an externally validated research benchmark.
Some study variables (large branching, semantic missing edges, external models,
multiple fault combinations) require future reviewed cases/adapters. The notebook
checks for direct impact, replacement, tampering and faults are small mechanics
experiments. Repeated-seed uncertainty and strong literature controls remain future work.

## Metric meanings
- Invalidated ratio: affected current certificates / initial current certificates.
- Recovery rate: new valid affected certificates / affected certificates.
- Recomputation ratio: rebuilt derived artifacts / initial derived artifacts;
  excludes raw dependencies and zero-cost verification receipts.
- Verification cost: configured cost of attempted certificate steps, including failed checks.
- Recovery/full costs: additive work units; certificate step includes verification
  plus issuance. Analysis overhead is included in selective recovery.
- Savings: `1 - selective/full`; undefined at zero full cost, may be negative.
  Compare only alongside safe useful completion. Partial/blocked work is cheaper.
- Latency: wall-clock update-to-policy/commit interval, measured only in future runs.
  Toy adapter full rebuild includes registry initialization; separately profile
  setup and verification for a fair publication comparison.
- Provenance overhead: exported journal bytes; storage bytes are not a latency result.
- Reliability: support-valid committed decisions plus useful task completion.
  Zero commits gives an undefined conditional rate and zero useful completion.
- Semantic probability ECE/Brier describe calibration against annotation, not truth.

Keep raw per-world records. Freeze tasks, costs, splits and model revision first;
do not tune from held-out results. Pair comparisons by world and report uncertainty.

