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
empty. It needs no processed research dataset. Each comparison uses unchanged
`nli-003` only and bypasses the Space/NLTK wrapper.

`config/pinned_smoke_models.json` stores immutable revisions, reviewed configurations,
class order and dataset hash. `cert_recovery.pinned_smoke` checks bindings before
heavy imports/downloads and requires both model-inference permission in config and
`manual=True`. In a future authorized free Colab session, install
`requirements-colab.txt` separately from the remote-client environment and enable
only the chosen smoke flag. Actual Colab/package/model execution remains untested.

A fresh preparation JSON captures exact pair, dataset/config/code/spec hashes,
runtime versions, reviewed model config and resolved revision before loading
weights. Strict loading checks reject missing or mismatched parameters. A second
new result records the weight digest, raw logits, all class probabilities, canonical
prediction, expected-label agreement, timing or errors. Probabilities use explicit
stable float64 softmax of captured logits and remain advisory. These direct runs
cannot retrospectively identify the Space's weights.

The general `config.yaml` model revision `main` belongs to guarded experimental
helpers; it is not the immutable smoke pin. Data/training/evaluation helper modules
remain for development and fixture tests. Their research execution and configured
DeBERTa performance have not been validated. No production, calibration or research
performance claim follows from the offline tests.
