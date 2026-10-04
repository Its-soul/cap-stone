# Model preparation — NOT YET RUN

**Prepared but intentionally not executed. Run manually in Google Colab.**

- Selected optional model: `cross-encoder/nli-deberta-v3-small` (Apache-2.0
  model card). Core needs no model. Its pretrained 3-class NLI output provides
  a deliberately limited entailment proxy baseline, not a causal detector.
- Input: context-bearing premise/claim text pair. Output: advisory probability
  of the annotated `depends_on` class after fine-tuning.
- Dataset: future human-reviewed dependency pairs; none collected/trained here.
- Fine-tuning: supervised full encoder fine-tuning, with a newly initialized
  binary classification head replacing the original 3-class NLI head.
- Pipeline: raw JSONL → clean/normalize → world split → paired tokenization →
  training → epoch checkpoints → validation-selected best model → held-out test.
- Hyperparameters: centralized in `config/config.yaml`; defaults are starting
  settings, not validated optimal settings. Seed, batches, rate, epochs, decay,
  accumulation, early stopping and checkpoint retention are configurable.
- Selection: validation macro-F1 by default; epoch evaluation/checkpointing
  and early stopping. Training never evaluates the held-out test split.
- Evaluation: accuracy, class precision/recall/F1, Brier and probability ECE.
  Threshold uses validation F1 with recall as tie-breaker; probabilities are
  **not assumed calibrated**. Report false-negative dependency errors separately.
- Colab: notebooks 02, 03, 04. All outputs empty and execution counts null.
- Integration: optional advisory analyzer interface only. Predictions never
  delete trusted edges, issue certificates, or bypass deterministic obligations.

Transformers is pinned to 4.57.1 to match the inspected Trainer API. Heavy imports
are lazy and guarded before import/download. Current `revision: main` is for
preparation: pin an immutable model commit before comparison. Run manifests
record resolved model revision and split/config hashes. Use a fresh checkpoint
directory for each model selection; resume explicitly when intended.

Source references (inspected 2026-10-04):
- https://huggingface.co/cross-encoder/nli-deberta-v3-small
- https://huggingface.co/cross-encoder/nli-deberta-v3-small/blob/main/config.json
- https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer
- https://huggingface.co/docs/transformers/v4.57.1/en/training

