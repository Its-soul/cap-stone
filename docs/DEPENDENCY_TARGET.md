# Binary dependency target

For a distinct candidate source S and named target artifact T, label **1** means
S is in T's explicitly declared transitive support ancestry. Label **0** requires
an exhaustive support contract proving absence. Incomplete context is unknown,
not an independent example. The target is a particular artifact, not an entire
premise/hypothesis string by default.

This is the implemented certificate engine's support/version binding contract.
The engine snapshots all declared ancestors and invalidates affected descendants
on version changes. It does not discover edges, minimize proofs or prune a read
because the output happens to stay unchanged. Predictions cannot change those
edges or authorize certificates; registered deterministic verifiers retain authority.

|Question|Meaning|Does it define this classifier's label?|
|---|---|---|
|Declared certificate support|Membership in the named target's declared ancestry|Yes|
|Value sensitivity|A particular intervention changes a computed value|No|
|Minimal necessary evidence|Removing one source makes every proof insufficient|No|
|NLI|Premise entails, contradicts, or leaves hypothesis undetermined|No|
|Similarity|Texts share vocabulary or topic|No|

Reading a source with coefficient zero still binds it. Two available OR witnesses
remain declared supports even when one suffices. Changing both sides of a
difference can leave its value unchanged while invalidating its certificate.
Conversely, mentioning a source in an observation does not make it support for a
separate numeric target. A contradictory checked statement cannot be certified
merely because its source is a declared dependency.

## Retrospective annotation findings

The original source-sum pilot and most candidate-v2 rules make support membership
and a selected value-changing intervention coincide. Their witnesses support those
particular examples, not a general equivalence between sensitivity and support.

- Sixteen independent `nli_context` mismatches retain label 0 under the recorded
  numeric-target scope. The hypothesis also repeats the candidate fact. Certifying
  that whole hypothesis asks a different dependency question; new inputs must name
  the scope explicitly. Eight high-confidence validation mismatches are a subset.
- Sixty-four historical approval negatives name required signatories without saying
  the list is exhaustive. Absence of another name does not establish label 0.
- Sixty-four negative `nli_trap` rows lack a target/support rule. Backup storage does
  not entail operational status and identical contents. The generator's entailment
  narrative is unsupported; no binary label inversion is established.
- Two candidate-v2 arithmetic false positives remain errors under explicit
  provisional exhaustive rules. Saved probabilities are advisory, not calibrated.

Original datasets, predictions and reports remain unchanged. Proposed scope
clarifications and exclusions are hash-bound, append-only review sidecars. All
reviews and new annotations are automated/provisional, pending mentor review.

## Prepared candidate v3, not executed

The new inputs declare target Q's exhaustive read supports and a **separate
observation check**. Shared context is repeated in both inputs; NLI concerns the
facts and that check. All six NLI/dependency combinations are meaningful under this
scope. Warranty claims are unknown when neither warranty state is asserted; hidden
generator state cannot determine NLI truth.

The mentor contrast collection has 64 rows: 11 per dependency label for entailment,
11 for contradiction and 10 for neutral. All are synthetic/provisional; none is
human-reviewed. Variants include bijective renaming, one support substitution with
a fixed cofactor, paraphrase, reordered facts and irrelevant context. Transformation
IDs bind exact parent/child pairs. Underdetermined inputs enter an abstention queue.

The frozen proposal has 256 training, 64 validation and 128 future-final rows in
28 groups, with 504 transformation links. Aliases, binary labels and positive support
positions are balanced. Related variants stay in one split; exact/normalized
duplicates and historical input reuse are rejected. Structural/template overlap is
reported explicitly for interpolation. Of the future-final rows, 64 interpolate,
32 use families held out from training but seen in validation, and 32 use families
held out from both training and validation. This is not a fully template-disjoint
evaluation. The observed 384-row stress and 96-row final suites are development data.

One principal intervention changes the **training distribution** to explicit scoped
NLI/support contrasts and unchanged-value read scenarios. The pinned base, seeded
fresh binary head, loss, 256-row training size, two epochs, optimizer settings and
validation-only checkpoint/threshold selection stay fixed. Distribution facets are
not separately isolated. No future-final predictions exist.

The frozen protocol reports accuracy, binary macro F1, class supports/confusions,
missed-dependency and false-alarm rates, category recall, worst scenario macro F1,
transformation consistency AND correctness, and unique coverage. Its primary
criterion is at least a 20 percentage-point reduction in entailment/independent
false alarms versus frozen v2 on the same future rows. Guards cover overall F1,
missed dependencies, all six categories, scenarios, renaming, support substitutions
and unchanged-value positive recall. A one-label category uses recall: fixed
two-class macro F1 would have a .50 perfect upper bound there.

The pinned tokenizer/truncation audit has now passed locally at the unchanged
maximum length 128. Before future execution: mentor review in new approval sidecars,
hash/environment checks, offline regressions and explicit authorization. Amend and
refreeze the proposal before training if these prerequisites reveal a required change.

Prepare a new private bundle without model loading:

```sh
python scripts/prepare_candidate_v3.py --output-dir <new-empty-private-review-directory>
```

The bundle contains reviews, immutable correction proposals, mentor CSVs, a local
HTML report, candidate JSONL files, protocol and freeze hashes, and a source/artifact
`phase_manifest.json`, checked by:

```sh
python scripts/validate_annotation_review.py <review-directory>
```

## Completed technical prerequisites; pending human review

The exact cache-only `DebertaV2TokenizerFast`, from DeBERTa revision
`fa2804872c3b4bd748f38c0185cc85775361e735`, was audited with Transformers 4.57.1.
Training and inference use full NFC/whitespace-normalized `premise`, `hypothesis`
strings in that order; they do not use `nli_projection` or append metadata.
V2 and v3 share this construction. V3's original text names Q, its scope, operation,
direct read supports, queried source and separate observation check. A closer review
found that source-root status was supplied by metadata/domain assumptions. An
exhaustive direct-read list alone cannot exclude a candidate from the ancestry of
a derived read support. Negatives need that additional condition made visible.

The original proposal is preserved. A versioned replacement,
`candidate-v3-data-v2-root-explicit` / `candidate-v3-plan-v2-root-explicit`, adds
"All sources have no supports." before the facts/check in BOTH strings. It changes
example wording within the declared distribution intervention. Labels, rules,
groups, transformation links, serialization, maximum length, base/head/loss/budget,
selection and acceptance guards remain unchanged. New hashes bind every replacement
row to its original hash; human review must bind the new protocol/data versions.

The audit covers 448 proposed rows plus 64 mentor contrasts and all 576 links.
Original maximum is 101 including three special tokens, with zero truncation;
256 negatives lack explicit ancestry closure, affecting 416 pair checks. The
root-explicit replacement maximum is 113; no rows truncate at 128.
Offset/token checks find no missing or lost decisive clauses. They include zero/
cancellation qualifiers, observation facts and checks. Both members of every
transformation retain their decisive text and encoded differences; batch-size-2
padding matches the row audit. Frozen v2 tokenizer IDs match for all 512 pairs.
These checks establish visible/retained declared information, not human approval
or model performance. Files and runtime versions are logged; no model was loaded.

Prepare a new cache-only prerequisite bundle in the existing model environment:

```sh
python scripts/prepare_v3_prerequisites.py --review-dir <frozen-review-directory> --output-dir <new-empty-bundle-directory>
python scripts/validate_v3_prerequisites.py <bundle-directory>
```

The bundle contains a compact mentor packet, all 64 contrast and 128 future-final
review rows, five historical examples including both arithmetic false positives,
an additive interpretation note with explicit subset denominators/coverage,
unchanged model-visible splits, immutable launch specification, templates and
source/artifact hashes. Reviewer fields remain blank. Review must approve the
target/scope contract and every required row, with identity, UTC date, rationale,
no unresolved ambiguity and exact artifact version/hash bindings.

Future execution uses this implemented command **only after new external approval
and authorization records exist**:

```sh
python scripts/run_candidate_v3.py --bundle <bundle-directory> --attempt-dir <new-attempt-directory> --execute --approval-file <new-human-approval.json> --authorization-file <new-training-authorization.json>
```

It requires authorization for training, validation inference and future-final
comparison, bound to the exact launch-spec hash. Templates and partial reviews
are rejected before creating attempts or importing model execution packages.
One exclusive launch claim prevents retries; errors/timeouts retain their files.
CPU uses four threads, one two-epoch/256-step training job, with a 1800-second
training limit and a separate 1800-second comparison limit. Paid compute and
network connections are blocked. GPU execution requires a separately reviewed plan.

The existing binary training pipeline retains validation-only checkpoint/threshold
selection. A saved immutable selection precedes future-final predictions; comparison
checks that freeze and all selected model/tokenizer hashes. Frozen v2 weights and
threshold .2 are unchanged. Launch specification/approval remain unapproved, all
preparation flags are false, and the command has not been executed with models.

Small grouped synthetic contrasts cannot establish general accuracy, calibration
or the research hypothesis.
