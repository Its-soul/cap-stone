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
