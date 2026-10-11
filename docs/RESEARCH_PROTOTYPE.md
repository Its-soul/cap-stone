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

## Status and next phase

IMPLEMENTED

- Four-agent orchestration, structured results, weighted voting and baseline.
- Six synthetic fault modes and preliminary validation/fault classification.
- Insurance demo, rule/support trace, cryptographic event DAG and tamper tests.
- Existing-engine selective recovery and a seeded offline experiment harness.

PARTIAL / PROTOTYPE

- CW-BFT direction: no rounds, network fault model or Byzantine quorum protocol.
- Calibration transform/reliability interfaces; authored confidence only.
- Rule provenance and content-addressed DAG, without causal identification or notarization.
- Model adapters without actual multi-LLM execution; synthetic robustness only.
- Coverage exclusions may require review; no complete insurance adjudication.

FUTURE WORK

- First add independently implemented model adapters behind the same validators,
  stronger full-claim baselines and ambiguous/noisy evidence cases.
- Fit confidence transforms on held-out role correctness data; freeze before testing.
- Evaluate genuine prompt-injection scenarios with multiple seeds and uncertainty
  intervals, including undetected errors and persistent recovery failures.
- Formalize quorum/fault assumptions, causal interventions and authenticated
  external checkpoints before evaluating the final 40% guarantee.

This milestone intentionally stops at executable, tested P0/P1 foundations and
small P2 interfaces. It is not a production insurance service or completed thesis.

## Validation record

The unchanged starting suite passed 271 tests, skipped one opt-in private V2
checkpoint test and passed 75 subtests. The final full suite passed **319 tests,
1 skipped, 75 subtests**, with zero failures: 48 new behavioral tests cover roles,
weighting, invalid confidence, calibration hooks, fault modes, backend isolation,
mandatory rules, event/support traces, tamper variants, selective recovery,
blocked reverification, baseline comparison and reproducible metrics.

The public preparation validator passed (one unexecuted notebook, two immutable
model profiles, disabled execution flags and valid documentation links).
Insurance, faulty-agent, tamper, recovery and four baseline-comparison demos ran;
the 100-trial framework ran offline. Tamper verification changed from true to
false after editing a historical payload. Original proposal text was checked
verbatim against the supplied request. No private-checkpoint inference ran.
