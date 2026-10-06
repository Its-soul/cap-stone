# Optional semantic clients and smoke preparation

The certificate engine needs no model. Repository-root `remote_model_client.py`
and `run_baseline.py` use optional Gradio/Hugging Face clients; install
`python -m pip install -e ".[remote]"` from the repository root.

## Remote workflow

`baseline_tests.json` contains six authored directional NLI smoke pairs, not a
benchmark. The runner records exact inputs, expected relations, raw/normalized
responses, independent transport/parsing/classification outcomes, end-to-end
latency, errors and available provenance in exclusive `results/baseline_*.json`
files. It stops on errors and has no inference retry loop. Success means a valid
service response, not agreement with the expected annotation.

Set optional credentials through `.env` using `.env.example`. Public access can
be anonymous; optional authentication needs a Hugging Face READ token. Credentials
are redacted from records. Endpoint/source/runtime/quota inspection is part of
`connect`; even that operation contacts the service. `--help` does not connect.
A future manually authorized call would use:

```sh
python run_baseline.py --profile nli-zero-gpu --allow-alternative-model --timeout 60
```

This command makes predictions; it is not a health check. No endpoint availability
or remaining quota is guaranteed. It uses an alternative wrapped service, never
claimed as execution of the configured `cross-encoder/nli-deberta-v3-small`.

## Reviewed profiles and interpretation

- `nli-zero-gpu`: Space `MridulSharma02/contradiction-detector`, endpoint
  `/contradiction_detector`, named inputs `sentence_a` and `sentence_b`.
  Reviewed Space commit: `f08cb1f21f447e96f7703b9c6b7c477ca2843f91`.
  The source may load local `fine-tuned-model` weights instead of fallback
  `cross-encoder/nli-MiniLM2-L6-H768`; loaded identity/revision remain unknown.
- `nli-cpu`: Space `tasksource/ModernBERT-zero-shot-nli`, endpoint `/process_input`,
  paired text with mode `Natural Language Inference`. Reviewed commit:
  `7660f8089995f96ee3d4d5a36bae734637275463`; source config names
  `tasksource/ModernBERT-base-nli`, without loaded-weight attestation.

Only the ZeroGPU profile maps `Consistent` to entailment. `Contradiction` maps to
contradiction. `Unrelated` can mean model neutral or the NLTK filter. Returned
rounded features indicate that filter when overlap is zero and WordNet similarity
is below 0.12. Those branch judgments are inferred from reviewed source, never
runtime-attested. Missing features leave the branch unknown; heuristic/unknown
output does not silently become model neutral. Unknown labels, malformed HTML and
invalid/nonfinite confidence values are rejected.

Schema-2 records retain the original response/label, confidence text and parsed
value, confidence origin, branch evidence, source/profile and separate configured,
fallback and verified-loaded identities. Rounded model softmax, heuristic scores
and unknown origins differ. Full probabilities are recorded only when supplied;
no confidence is assumed calibrated. Latency includes network, queue and service
processing. A safe recorded fixture retains the high-confidence pizza/bicycle
mismatch; neutral remains its authored annotation. The cause and loaded weights
are unresolved.

## Disabled direct-model notebook

`notebooks/02_baseline_model.ipynb` is a usage example for separate pinned DeBERTa
and MiniLM comparisons, with all execution/installation flags false and outputs
empty. It needs no processed research dataset. Each notebook comparison uses all
six unchanged pairs, including `nli-003`, and bypasses the Space/NLTK wrapper. The
library's default remains a single-case smoke; `all_cases=True` explicitly opts
into six.

`config/pinned_smoke_models.json` stores immutable revisions, reviewed configurations,
class order and dataset hash. `cert_recovery.pinned_smoke` checks bindings before
heavy imports/downloads and requires both model-inference permission in config and
`manual=True`. In a future authorized free Colab session, install
`requirements-colab.txt` separately from the remote-client environment and enable
only the chosen smoke flag. Colab execution remains unverified; each local run
records its own actual runtime and status. Empty notebook outputs or the profile's
preparation status do not describe a separately executed run.

A fresh preparation JSON captures exact pair, dataset/config/code/spec hashes,
runtime versions, reviewed model config and resolved revision before loading
weights. Strict loading checks reject missing or mismatched parameters. A second
new result records the weight digest, raw logits, all class probabilities, canonical
prediction, expected-label agreement, timing or errors. Probabilities use explicit
stable float64 softmax of captured logits and remain advisory. These direct runs
cannot retrospectively identify the Space's weights.

The general `config.yaml` revision `main` is an inactive default. Executed model
pipelines require an immutable revision. The bounded pilot copies the reviewed
DeBERTa pin into its private configuration; public execution flags remain false.
Its initial two-class dependency head replaces the three-class NLI head and is
trained from a seed, never described as a resumed or already trained model.

## Pilot training and evaluation

`scripts/run_pilot.py --execute` reuses `prepare_data`, `run_pretrained_baseline`,
`run_finetuning`, `run_model_evaluation` and the certificate workflow evaluator.
It runs one two-epoch training configuration, batch size 2, maximum pair length
128, learning rate 2e-5, weight decay .01 and seed 42. CPU uses float32; automatic
device placement also supports CUDA. A tokenization audit rejects truncation of
training/validation task context. Training logs every optimizer step, validates
each epoch and selects the best checkpoint by validation macro F1 with the existing
early-stopping settings. It saves actual selected weights/tokenizer, Trainer state,
loss curves, split/config bindings and digests. Resume must be explicit and point
to a real checkpoint with model, optimizer, scheduler and Trainer state, within
the same bound run; a nonempty destination cannot become an accidental fresh run.

The dependency threshold is selected on validation and persisted before held-out
inference. Evaluation verifies the config, splits and selected weight bytes and
uses that frozen threshold. No retuning follows held-out inspection. The comparison
uses DeBERTa NLI entailment probability as a proxy on the same dependency rows,
with its own validation threshold. That proxy has a different training objective;
its aggregate binary scores are not calibrated dependency probabilities. Original
three-class logits/probabilities are also retained. Confusion matrices, both classes'
precision/recall/F1, macro F1, false-negative dependency counts, Brier/ECE summaries
and error examples are descriptive measurements of a tiny synthetic sample.
Existing model-selection/evaluation records cannot be overwritten or silently
reinferred; use offline report replay or a new result directory. Completed training
selections likewise cannot become accidental resume runs.

Direct immutable loading records base weight digests, verified configurations,
label order and loading information; only the expected classifier shape change is
allowed during initial fine-tuning. Model confidence still cannot authorize a
certificate or override deterministic verification. Direct findings cannot identify
the historical Space's unverified local/fallback weights.

The implementation follows the versioned [Transformers Trainer contract](https://huggingface.co/docs/transformers/v4.57.1/trainer).
Generated private reports distinguish actual execution from preparation and say
**PRELIMINARY / EXPERIMENTAL — not final or production results.**

The controlled candidate workflow retains the same pinned pretrained starting
model and replaces its three-class NLI head with a fresh seeded binary dependency
head. It preserves the original trained checkpoint as a comparator. Checkpoint
selection uses validation macro F1; threshold selection uses validation positive
F1, recall and then the smallest candidate. Stress/final scores cannot select a model.
New selections record the initial head digest, actual optimizer steps and every
saved model/tokenizer file digest. Historical runs without those records retain
their provenance limitations; post-hoc file checks do not attest past runtime loading.

The NLI proxy has its own validation-selected threshold and remains a different
objective. Constant and identifier diagnostics fit training only; coverage and the
training-prior fallback are explicit. Constant fallback behavior for unseen aliases
is not evidence that a model avoided source-name shortcuts. Saved classification
and transformation metrics are computed separately, with exact input alignment.

## Current annotation review and v3 proposal

The [dependency contract](DEPENDENCY_TARGET.md) defines declared transitive support
for a named target, rather than sensitivity or NLI. Saved-prediction review covers
544 rows, including 16 scope-dependent `nli_context` mismatches, 49 stress regressions
and all 42 stress improvements. Eight high-confidence validation errors are a
subset of the 16. Missing exhaustive context prevents confirming 128 historical
negative annotations. No causal memorization claim follows from these observations.

V3 is preparation only: new grouped data and a frozen distribution intervention;
the base, binary head/loss, budget and validation selection stay fixed. All flags
remain false. The original pairs fit within 101/128 tokens, but negative transitive
dependency judgments lacked an explicit source-root condition. A new version adds
"All sources have no supports." to both strings. All 512 repaired pairs fit within
113/128 tokens, retain decisive clauses and match frozen v2 token IDs.
Mentor approval and manual authorization precede any job. No v3 inference, training or future-final predictions
have run. Historical checkpoint and source hashes remain intact. Probabilities
never establish certificate validity, and annotation proposals never modify edges.

The disabled `run_candidate_v3.py` launcher reuses `run_finetuning`, blocks network,
checks external review/authorization, binds one fresh attempt and freezes validation
selection before comparing the new future-final set. It preserves the exact base,
fresh binary head, loss, two-epoch budget and selection policy. The launch specification
is proposed, not approved; none of these future model stages ran during preparation.
