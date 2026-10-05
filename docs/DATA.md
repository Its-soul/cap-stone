# Data formats

The core works with caller-supplied JSON payloads and explicit version-bound
supports; it does not fetch sources or discover dependency edges automatically.

`baseline_tests.json` contains six fixed authored NLI pairs. Each has `id`,
`input.premise`, `input.hypothesis` and `expected_label` (entailment, contradiction
or neutral). Premise comes first. These are smoke fixtures, not a research dataset.
The recorded case-003 HTML fixture is safe test data, not a new inference result.

The guarded data helpers accept JSONL rows shaped like:

```json
{"pair_id":"p1","world_id":"world-1","premise":"Supplier A is eligible.","hypothesis":"The decision uses A's eligibility.","label":1,"source":"authored-example"}
```

Binary dependency labels differ from NLI labels: `1` means required support in the
stated task, `0` means independent. Entailment/similarity alone do not establish
required support. `data/examples/` contains authored format examples only.

Cleaning normalizes Unicode/whitespace and rejects conflicting or invalid rows.
Grouped splits separate worlds and reject repeated pairs across splits. Split
manifests bind hashes; model selection must not use held-out test results.
Configured raw/processed data paths, results and checkpoints are ignored by Git.
Supply reviewed, licensed inputs explicitly; no data preparation or downloads
happen on import. All execution flags remain false by default.

## Controlled synthetic pilot

`cert_recovery.pilot.synthetic_pairs` reuses the implemented arithmetic workflow.
Each target claim sums its declared transitive source ancestry. A source is labeled
`depends_on` precisely when it is in that ancestry; increasing it by one must change
the rebuilt claim by one. An independent arithmetic oracle checks the same rule in
offline tests. Both independent and cascade topologies are included. The pair text
names the candidate source/value and the claim's exhaustive source list, giving
explicit task context. These labels mean declared support under this contract;
they do not establish real-world causal necessity or general semantic dependency.

There are 12 phrasing families, four related worlds per family and one positive/
one negative pair per world: 96 rows. The pilot config groups by `template_family`,
so every related world and every use of that template stays in one split. Normalized
duplicates cannot span split groups. Seed 42 yields 64 training, 16 validation,
16 held-out rows, balanced within each split. Raw private records retain the label
derivation, declared bindings and intervention witness. The existing cleaner keeps
negation and numbers. Neither the six authored NLI pairs nor their expected labels
are used for fine-tuning. The data and split manifest are saved before model loading.

The source-identifier contingency audit exposes a pilot limitation: chosen claims
are Q0/Q1, so D2 is never a required source. Disjoint families prevent exact template/
world leakage but do not remove this structural naming shortcut. Record the audit
with results; do not rewrite the frozen pilot or tune against its held-out examples.
A new evaluation set should vary source aliases/roles and include reviewed context.
