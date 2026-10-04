# Data

No dataset preparation or large dataset run was executed. No downloads are automatic.

## Semantic dependency annotation
Provide JSONL at the configured `raw_pairs` path. Each row has:

```json
{"pair_id":"p1","world_id":"world-1","premise":"Supplier A is eligible.","hypothesis":"The decision selects A using its eligibility.","label":1,"source":"human-reviewed-example"}
```

`label=1` means the annotated claim uses the premise as required semantic support;
`0` means independent in the stated task. Put the relevant task context in the
text. Similarity, entailment and causal necessity are different concepts.
Have annotators record disagreements and independently adjudicate labels.

Cleaning normalizes Unicode/whitespace, rejects missing/invalid fields and
conflicting labels, removes same-world duplicate pairs, and rejects pairs shared
across worlds. Group connected worlds before splitting shared templates/content.
Train/validation/test split by world, never random rows. Fractions approximate
group counts; label balance is inspected manually. Manifests bind split hashes.
Model selection and thresholds use validation only; freeze before viewing test.

The small files in `data/examples/` are authored format examples, not a dataset
or research evidence. Copy them to configured paths only for manual Colab setup
checks. Use genuinely reviewed domain worlds for experiments.

## Workflow cases
Prepared arithmetic adapter uses `world_id`, `topology` (`independent`/`cascade`),
integer-valued `sources`, and a `change` with source index, kind (`update`,
`remove`, `untrusted`), and optional new value. Separate source arithmetic is
the oracle; no gold label is passed to a semantic model/verifier.

This toy adapter tests the mechanics. Procurement/eligibility rules, external
trace datasets and real fault-injection adapters remain future work. Public NLI
datasets can provide transfer learning; their labels are not dependency gold.
Review licensing before importing or redistributing data.

