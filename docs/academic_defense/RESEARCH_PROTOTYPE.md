# Insurance research prototype milestone

This is a bounded, offline, synthetic research milestone. The original
[project description](PROJECT_PROPOSAL.md) remains the final research direction.
No formal BFT guarantee, calibrated model confidence, 40% robustness guarantee,
complete causal inference, regulatory certification or immutable external
notarization is claimed.

## Inspection and preservation

Inspected the source/API surface, recovery implementation, certificate lifecycle,
dependency DAG, crypto/audit journal, advisory adapter and remote client,
model/data/evaluation pipelines, V2/V3 guards, tests, scripts, configuration,
documentation, fixtures and notebook cell/execution inventories. Git was initially
clean. Tracked and ignored inventories and the private archive's preservation
guidance were reviewed; private checkpoints and evidence were not altered.

Before this milestone, multi-agent adjudication, confidence-weighted consensus,
agent-adversarial evaluation and insurance adjudication were MISSING. Causal/rule
provenance and cryptographic DAG provenance were PARTIAL. Recovery was STRONG.
The review needs one qualification: `metrics.py` already supplied Brier/ECE
calculations, and existing evaluation code already exercised synthetic dependency
faults and audit integrity. Those were not agent-adversarial evaluations or fitted
confidence calibration. The dependency contract also already had explicitly
declared structural intervention witnesses, not general causal inference.

The original engine, graph, certificates, linear journal, semantic advisory
boundary, all existing tests, configuration, V2/V3 protocols and historical results
are unchanged. New code calls the existing facade and registered recipe/verifier
interfaces. No credentials, paid compute, model downloads, training, commits or
pushes are needed.

## Implemented flow and code map

`src/cert_recovery/adjudication/` contains:

| Module | Responsibility |
|---|---|
| `domain.py` | Typed synthetic claim, four role contracts/prompts, backend protocol, deterministic role agents, six output fault modes |
| `consensus.py` | Structured output validation, trust assessment, normalized vote inputs, confidence weighting and majority baseline |
| `ledger.py` | Canonical payload/event hashing, multiple parent hashes, aggregate tip commitment, verification and traversal |
| `workflow.py` | Agent execution, validation, voting, deterministic safety gate, certificate issuance and registered selective rebuilders |
| `experiments.py` | Authored cases/labels, same-output baseline comparison, seeded fault sampling, per-trial and aggregate measurements |

`scripts/run_insurance_prototype.py` runs demos or experiments without model
imports. `tests/test_insurance_adjudication.py` exercises behavior, not class
existence. Package imports perform no inference or downloads.

Policy checks coverage, waiting period and the INR amount/limit; evidence checks
documents and invoice consistency; risk checks duplicate/provider flags;
compliance checks identity and consent. Each role has different inputs, rules,
prompts, outputs and confidence. These are four logical agents using one
deterministic backend implementation. The role-specific predicates differ;
they are not four copies of the same vote. `AgentBackend.execute` permits a
different adapter per role later, but no independent-model result is claimed.

Every record includes agent identity, role, structured raw decision/confidence,
evidence, reasoning summary, rule IDs, exact version/hash input binding,
assessment, and a provenance reference. Malformed results and backend exceptions
are retained as failures and receive zero voting weight. Returned evidence is
checked against trusted fixture facts. Reasoning prose is retained but not
semantically verified; malicious intent is never inferred.

Validation distinguishes rule-conflicting `INCORRECT` output, evidence/rule/input
mismatch or overconfident invalid `SUSPICIOUS` output, and malformed/execution
`CONFIRMED_FAILURE`. Scores are intentionally binary (valid=1, fault=0, or the
reverse), not learned probabilities of maliciousness. Fault modes are incorrect
decision, inflated confidence, misleading evidence, contradictory rule citation,
malformed output and execution failure. They are output corruptions, not live
prompt-injection attacks.

## Consensus and authority

Weight = transformed confidence × reliability × validation score × (1 − fault score).
Scores are summed per decision with deterministic ordering and `math.fsum`.
A unique highest score wins; a tie, empty input, or zero total weight yields REVIEW.
Duplicate identities and nonfinite/out-of-range scores are rejected. Reliability
defaults to one. Validation currently excludes faulty outputs completely.

Confidence really changes voting outcomes: tests include two approvals at 0.2
each losing to one rejection at 0.9, then winning after one approval rises to 0.8.
The optional calibration transform and reliability/fault weights are extension
points. Default authored confidences are explicitly uncalibrated. Shannon entropy
over weighted vote shares is recorded; it is **not latent model entropy**. No
temperature-scaling fit or calibration performance was measured. Existing
`metrics.calibration_metrics` can evaluate held-out binary confidence data later.

Role votes are local recommendations, not interchangeable complete claim judgments.
The global mandatory-rule outcome is REJECT for a coverage exclusion, REVIEW for
missing evidence/risk/control conditions, otherwise APPROVE. A weighted proposal
matching that outcome passes; disagreement yields REVIEW. Consequently confidence
cannot override a mandatory rule. An ordinary policy exclusion often produces
three local approvals and one rejection, so this prototype conservatively returns
REVIEW rather than completing an automatic rejection. This is a measured limitation.

The registered round verifier independently recomputes assessments, votes and the
mandatory gate and checks their event bindings before issuing a certificate.
Final artifacts depend on that certificate. Certification attests to following
the declared computation, **not** real-world correctness of source facts, coverage,
model truthfulness or medical/legal/insurance compliance. V2 model confidence
retains its separate advisory-only role.

## Two linked provenance structures

The existing artifact DAG holds role-specific fact sources → agent assessments →
round → verification receipt/certificate → final decision, all bound to exact
versions/hashes. Payload metadata identifies source, node type, rules, evidence,
agent and decision impact. `system.provenance(final_ref)` provides the complete
declared rule/support trace. This is structural rule provenance, not identified
causal effects or a full SCM.

The added event DAG records input, evidence access, execution/prompt, raw output,
validation, fault assessment, vote, consensus, final decision and recovery. Each
event commits to a canonical payload hash, timestamp, ID/type, agent identity and
sorted parent event hashes. Consensus merges the four vote branches. A checkpoint
commits to event count and the sorted current tips; final records bind the engine
certificate hash. The engine's original linear audit chain remains intact and
separately verifiable; recovery events also record its checkpoint.

`EventDAG.verify(records, expected_checkpoint)` checks payload/event hashes,
parent existence/order, IDs and the aggregate commitment. Tests detect payload,
parent, timestamp, hash, order, deletion and truncation tampering. `trace(hash)`
walks contributors. Identical records with a fixed clock hash identically; real
runs include timestamps and are not expected to have identical hashes.

Persist checkpoints separately if they must serve as trusted anchors. An attacker
able to rewrite the entire history and replace its checkpoint can rehash it; tests
explicitly demonstrate this limitation. There is no signature, external timestamp
authority, durable database, distributed replication or production immutability.

## Selective recovery

An explicit agent failure invalidates that agent artifact, round, certificate and
final decision. The existing engine rebuilds only those artifacts, reverifies the
round and supersedes the old certificate. Other agent artifacts, hashes, versions
and execution counts stay unchanged. A source change rebuilds only the role(s)
that consume the changed facts. Old decisions cannot pass the commit gate.

The recovery demo starts with a failed risk agent, clears the injected failure,
and finishes with execution counts policy=1, evidence=1, risk=2, compliance=1.
The failure is not silently erased: prior records and certificate history remain.
Recovery-start ancestry connects the new output back to the old final event;
completion records the outcome and execution counts. Clearing a synthetic fault
stands in for repairing/replacing an agent backend. Persistent backend faults
remain excluded even when the pipeline can rebuild a safe REVIEW/other result.

## Measured synthetic experiment

Commands (from the repository root):

```sh
python scripts/run_insurance_prototype.py --output results/new_demo.json
python scripts/run_insurance_prototype.py --experiments --trials 20 --seed 42 --output results/new_trials.json
python -m pytest tests -q
```

Output files use exclusive creation. Existing files are never overwritten.
The milestone run on 2026-10-06 used 20 trials per requested rate, 100 trials
total, seed 42, and five cyclic authored cases (approve, coverage exclusion,
missing documents, duplicate claim, missing identity verification). Labels are
kept outside the agent/verifier inputs. Each agent independently receives a
Bernoulli fault selection, then one of the six corruption modes. Four agents
cannot realize 10%, 20%, 30% or 40% exactly within one trial, so realized aggregate
rates and affected-agent counts are reported rather than rounding and relabelling.

| Requested rate | Realized rate | Affected / 80 | Majority accuracy | Validated weighted-only accuracy | Proposed + gate accuracy |
|---|---|---|---|---|---|
| 0% | 0% | 0 | 20% | 20% | 80% |
| 10% | 8.75% | 7 | 25% | 20% | 80% |
| 20% | 21.25% | 17 | 15% | 20% | 80% |
| 30% | 33.75% | 27 | 35% | 30% | 85% |
| 40% | 32.5% | 26 | 25% | 30% | 85% |

Detection recall was 100% for injected outputs; false suspicion was 0%. Recovery
success was 100% in 5, 12, 18 and 16 fault-bearing trials respectively (undefined
at 0%, where recovery was unnecessary). These measurements reflect deliberately
detectable mutations, trusted rule-complete facts, known injected agent identities
and repaired deterministic backends. They do not establish general fault detection.
All trial decisions, fault identities/modes, actual rates, recovery counts and
metric denominators are exported for reconstruction.

Local retained evidence is in `results/insurance_milestone_final_seed42.json`
and `results/insurance_milestone_final_demo.json` (ignored generated files).
Earlier milestone captures were preserved; the final trial records reproduced
the initial run exactly. Rerun with a new output filename to retain another run.

The baseline uses the same raw outputs but no validation/gate. It is intentionally
weak because it votes over role-local checks. Most accuracy improvement comes
from the deterministic gate, **not demonstrated superiority of weighting**;
weighted-only results make that clear. Higher fault rates occasionally improve
accuracy by excluding locally approving voters in rejection cases; that is a
voting artifact, not beneficial adversarial robustness. A stronger full-claim
baseline is future work. The requested 40% run neither realizes exactly 40% nor
proves the abstract's claim. Twenty repeated small synthetic cases per setting
are insufficient for statistical or model-quality claims.

The four fixed comparison scenarios approve the clean claim in both systems
with zero or one faulty agent. With three high-confidence corrupted votes, majority
returns REJECT and validated weighting returns APPROVE. That 3/4 scenario is an
illustration of rule-based exclusion, not a 75% Byzantine tolerance claim.

## Ablation study and mechanism attribution

To determine which mechanisms account for observed performance and test whether confidence weighting justifies its complexity, an offline factorial ablation study was conducted across both historical (5 cases, 75 trials) and extended suites (10 cases, 150 trials) across three verified seeds (`[42, 101, 2024]`, correcting the errant documentation claim of seed 137) and five fault rates (0.0 to 0.4).

The study explores the complete 2x2x2 factorial space of (Majority vs Weighting) x (Validation OFF vs ON) x (Gate OFF vs ON), plus the deterministic rules-only reference:

- **Arm A (Majority only)**: Unweighted consensus, validation OFF, gate OFF.
- **Arm B (Majority + Mandatory Gate)**: Unweighted consensus, validation OFF, gate ON.
- **Arm C (Confidence weighting only)**: Confidence-weighted consensus, validation OFF, gate OFF.
- **Arm D (Weighting + Detection)**: Confidence-weighted consensus, validation ON, gate OFF.
- **Arm E (Full prototype)**: Confidence-weighted consensus, validation ON, gate ON.
- **Arm F (Deterministic rules only)**: Pure deterministic rule evaluation bypassing agents and consensus (reference oracle).
- **Arm G (Majority + Validation + Gate)**: Unweighted consensus, validation ON, gate ON.
- **Arm H (Majority + Validation, no gate)**: Unweighted consensus, validation ON, gate OFF.
- **Arm I (Weighting + Gate, no validation)**: Confidence-weighted consensus, validation OFF, gate ON.

### Empirical results (Extended Suite: 10 cases, 150 trials)

| Arm | Method | 3-Class Accuracy | Decisive Coverage | Decided Accuracy | Unsafe Approvals | Required Reviews | Additional Deferrals |
|---|---|---|---|---|---|---|---|
| **A** | Majority only | 22.7% (34/150) | 93.3% (140/150) | 22.1% (31/140) | 107 | 3 | 7 |
| **B** | Majority + Mandatory Gate | 70.7% (106/150) | 20.7% (31/150) | 100.0% (31/31) | **0** | 75 | 44 |
| **C** | Confidence weighting only | 23.3% (35/150) | 100.0% (150/150) | 23.3% (35/150) | 109 | 0 | 0 |
| **D** | Weighting + Detection (no gate) | 22.7% (34/150) | 98.7% (148/150) | 22.3% (33/148) | 115 | 1 | 1 |
| **E** | Full prototype (Weight + Val + Gate) | 72.0% (108/150) | 22.0% (33/150) | 100.0% (33/33) | **0** | 75 | 42 |
| **F** | Deterministic rules only (Reference) | **100.0% (150/150)** | **50.0% (75/150)** | **100.0% (75/75)** | **0** | **75** | **0** |
| **G** | Majority + Validation + Gate | 70.7% (106/150) | 20.7% (31/150) | 100.0% (31/31) | **0** | 75 | 44 |
| **H** | Majority + Validation (no gate) | 25.3% (38/150) | 93.3% (140/150) | 22.1% (31/140) | 109 | 7 | 3 |
| **I** | Weighting + Gate (no validation) | 73.3% (110/150) | 23.3% (35/150) | 100.0% (35/35) | **0** | 75 | 40 |

*Note on Arm D decided accuracy*: Previously reported figures conflated 3-class correct numerator (34) with decided cases denominator (148). The correct decided accuracy is strictly $33 / 148 = 22.30\%$.

### Single-factor matched differences (changing one factor at a time)

1. **Gate Effect (Gate ON vs Gate OFF)**:
   - Majority unvalidated: B (70.7%) vs A (22.7%) -> **+48.0%** accuracy, unsafe 107 -> 0 (-107)
   - Majority validated: G (70.7%) vs H (25.3%) -> **+45.3%** accuracy, unsafe 109 -> 0 (-109)
   - Weighted unvalidated: I (73.3%) vs C (23.3%) -> **+50.0%** accuracy, unsafe 109 -> 0 (-109)
   - Weighted validated: E (72.0%) vs D (22.7%) -> **+49.3%** accuracy, unsafe 115 -> 0 (-115)
   - *Mean Gate Effect*: **+48.17%** accuracy improvement across all consensus/validation settings; completely eliminates unsafe approvals.
2. **Validation Effect (Validation ON vs Validation OFF)**:
   - Majority ungated: H (25.3%) vs A (22.7%) -> **+2.67%** accuracy, unsafe +2
   - Majority gated: G (70.7%) vs B (70.7%) -> **0.0%** accuracy, unsafe 0
   - Weighted ungated: D (22.7%) vs C (23.3%) -> **-0.67%** accuracy, unsafe +6
   - Weighted gated: E (72.0%) vs I (73.3%) -> **-1.33%** accuracy, unsafe 0
   - *Interpretation*: Under gating, validation filtering slightly reduces accuracy (-1.33%) because zeroing faulty agent weights when valid agents disagree creates ties that fall back to REVIEW.
3. **Consensus Weighting Effect (Weighting vs Majority)**:
   - Unvalidated ungated: C (23.3%) vs A (22.7%) -> **+0.67%** accuracy, unsafe +2
   - Unvalidated gated: I (73.3%) vs B (70.7%) -> **+2.67%** accuracy, unsafe 0
   - Validated ungated: D (22.7%) vs H (25.3%) -> **-2.67%** accuracy, unsafe +6
   - Validated gated: E (72.0%) vs G (70.7%) -> **+1.33%** accuracy, unsafe 0
   - *Interpretation*: Marginal contribution of confidence weighting under the full pipeline (Arm E vs Arm G) is **+1.33 percentage points**.

### Negative finding: Rules-only baseline outperforms multi-agent pipeline
- **Method F (Deterministic rules only)** achieved **100% accuracy** on all decided cases and **50.0% decisive coverage** with **0 false reviews**.
- By comparison, the full multi-agent prototype (Method E) achieved only **72.0% accuracy** and **22.0% coverage**, inducing **42 false reviews** (additional deferrals) on valid rejection scenarios.
- **Root cause: Role-fusion category errors.**
  In coverage exclusion scenarios (e.g. cosmetic surgery), non-policy agents (evidence, risk, compliance) correctly approve their specific domain checks (documents valid, no fraud, identity confirmed). Consequently, the raw agent vote is 3-to-1 in favor of APPROVE. The mandatory-rule gate detects the disagreement between the policy exclusion (REJECT) and the consensus proposal (APPROVE) and safely defers to `REVIEW`. Pure deterministic rules evaluate policy exclusions globally without consensus overhead, achieving decisive rejection without false reviews.

## Provenance and selective recovery benchmarking

A benchmark comparing selective recovery against full recomputation was evaluated across four operational failure scenarios. The full recomputation harness was repaired to eliminate duplicate recovery: both methods begin from equivalent initial state, and full recomputation cleanly recomputes strictly 4 agent artifacts in a single recovery pass:

| Scenario | Rebuilt Agents (Selective) | Rebuilt Agents (Full) | Preserved Agents (Selective) | Selective Cost | Full Cost | Saved Executions | Speedup Ratio (Mean) |
|---|---|---|---|---|---|---|---|
| **Single Agent Failure (`risk`)** | 1 | 4 | 3 (v1 preserved) | 4.0 | 7.0 | 3 | ~1.27x |
| **Single Fact Change (evidence)** | 1 | 4 | 3 (v1 preserved) | 4.0 | 7.0 | 3 | ~1.21x |
| **Multiple Affected Branches (policy + evidence)** | 2 | 4 | 2 (v1 preserved) | 5.0 | 7.0 | 2 | ~1.12x |
| **Repeated Changes Before Recovery** | 2 | 4 | 2 (v1 preserved) | 5.0 | 7.0 | 2 | ~1.21x |

### Timing and slowdown disclosure
- **Mock backend characteristics**: Agent execution in this offline prototype utilizes instantaneous in-memory Python mock backends (~0.01ms per execution).
- **Harness repair disclosure**: Previously, full recomputation called `update_claim()` prior to full invalidation, which triggered selective recovery before full recovery. This double-counted executions (reporting 5 and 6 executions) and inflated runtime.
- **Timing reality**: With clean single-pass recomputation, speedup on mock backends is modest (~1.1x to 1.3x) because wall-clock time is dominated by DAG traversal, serialization, and cryptographic hashing. Under system noise, selective and full recovery distributions can overlap. Genuine wall-clock acceleration requires non-trivial external model latency.

### Cryptographic tamper verification and trust model
- Event DAG verification was tested against payload modifications, parent deletions, event reordering, and truncated histories.
- All tampering attempts are detected when evaluated against a **retained external checkpoint anchor**.
- **Known limitation**: If an attacker controls both history and the anchor generation process, the Event DAG can be rehashed without detection. The external anchor is the sole cryptographic trust root.

## Model-backed agent integration

A model-backed adapter (`ModelAgentAdapter`) was implemented for the regulatory compliance role with a frozen prompt and output schema contract targeting `Qwen/Qwen2.5-0.5B-Instruct`.

- **Evaluation scope**: 11 diverse compliance test cases evaluated against the deterministic role.
- **Execution status**: Genuine live GPU model execution remains **NOT RUN** due to local offline environment constraints. Evaluation executed exclusively via the offline schema/prompt adapter simulation.
- **Empirical observation disclosure**: Reported 100% agreement and latency metrics (5.47ms vs 0.01ms) are **mock simulation outputs** produced by the offline fallback adapter. They do not represent genuine LLM inference performance.
- **Calibration disclosure**: Self-reported confidence values (0.88 on approve, 0.72 on review) are synthetic mock strings. Live temperature scaling or Platt scaling remains **NOT RUN**.
- **Error containment**: Injected JSON syntax errors, missing schema fields, and execution timeouts were contained: assessed as `CONFIRMED_FAILURE`, allocated 0.0 voting weight, and prevented from corrupting consensus.

## Status and component breakdown

### IMPLEMENTED
- Four-agent orchestration, typed schemas, validation filtering, and weighted consensus.
- Deterministic mandatory-rule safety gate.
- Full 8-arm factorial + rules reference ablation harness and CLI runner (`scripts/run_insurance_ablation.py`).
- Event DAG with multi-parent hashing, tip commitment, and tamper verification.
- Selective recovery engine rebuilding only invalidated nodes while preserving valid nodes at v1.
- Repaired recovery benchmark with single-pass full recomputation and repeated timing distributions.
- Model agent adapter interface with structured JSON decoding and fault containment.
- Reproducible checkpoint bundle binding git revision, environment, and raw records.

### PARTIAL / PROTOTYPE
- Single model-backed role adapter (`compliance`); policy, evidence, and risk remain deterministic.
- Model execution evaluated in offline simulated adapter mode; live weights unexecuted.
- Authored and self-reported confidence values are uncalibrated; calibration fitting deferred.
- Structural rule provenance without general SCM causal effect identification.

### BLOCKED / UNAVAILABLE
- Live GPU multi-LLM execution (local GPU and downloaded weights unavailable).
- External trusted timestamp authority or distributed blockchain notarization.
- Formal Byzantine fault-tolerant quorum protocol (no asynchronous network model).

### FUTURE WORK
- Implement independent model backends across all four roles and evaluate under live local LLM inference.
- Formulate whole-claim agent prompts to eliminate role-fusion category errors and false reviews on exclusions.
- Fit empirical calibration maps on held-out agent prediction data.
- Formalize external anchor distribution and authenticated checkpointing.

## Validation record and reproducible checkpoint

The complete test suite passed **352 tests, 1 skipped, 75 subtests passed** with zero failures across the entire repository:
- `tests/test_harness_repaired_validation.py`: 7 tests covering metric numerators/denominators, matched settings, recovery counting bounds, source completeness, and overwrite guards.
- `tests/test_insurance_adjudication.py`: 48 tests covering consensus, roles, fault modes, recovery, and event DAG.
- `tests/test_ablation_study.py`: 8 tests covering 9-arm ablation methods, metric correctness, and output persistence.
- `tests/test_provenance_recovery_integrity.py`: 9 tests covering selective recovery cost savings and Event DAG tamper detection.
- `tests/test_model_agent.py`: 9 tests covering model adapter parsing, prompt construction, and error containment.
- `tests/test_v2_execution.py`, `tests/test_core.py`, etc.: 271 baseline tests.

### Checkpoint inspection commands

Two checkpoint bundles are retained:
1. **Preserved Original Checkpoint**: `_private/runs/adjudication_milestone_20261012/` (preserved historical capture).
2. **Repaired Milestone Checkpoint**: `_private/runs/checkpoint_adjudication_repaired_20261012/`:
   - `manifest.json`: Binds git commit `1dc61fd`, platform, environment, model identities, case suites, verified seeds (`[42, 101, 2024]`), and SHA-256 hashes of all raw outputs and source snapshots.
   - `source_snapshot/`: Byte-identical copies of all relevant source and test files, including untracked additions.
   - `source_manifest.json`: Relative paths and SHA-256 hashes of all source snapshot files.
   - `execution_environment.json`: Exact Python version (`3.14.6`), OS, and package inventory.
   - `local_changes.diff`: Tracked git diff and untracked file status.
   - `config_and_prompts.json`: Frozen role contracts, prompts, verified seeds (`[42, 101, 2024]`), and fault schedules.
   - `raw_records/`: Preserved historical raw JSON results and repaired factorial/benchmark outputs.
   - `ANALYSIS.md`: Detailed validation analysis.

To reproduce or inspect the offline milestone:

```sh
# Run full offline test suite
python -m pytest tests -q

# Run public preparation verification
python scripts/validate_preparation.py

# Run ablation study on extended suite (150 trials, 3 verified seeds)
python scripts/run_insurance_ablation.py --suite extended --seeds 42 101 2024 --output results/ablation/my_ablation.json

# Run recovery benchmark
python -c "from cert_recovery.adjudication.recovery_benchmark import run_recovery_benchmark; print(run_recovery_benchmark())"
```
