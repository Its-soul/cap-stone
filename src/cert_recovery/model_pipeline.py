from __future__ import annotations

import json
import re
from pathlib import Path
from time import perf_counter

from .config import require_manual, resolve_path
from .crypto import sha256
from .data_pipeline import load_verified_splits, write_json
from .metrics import binary_metrics, calibration_metrics
from .pinned_smoke import sha256_file, validate_label_order


def _loaded_provenance(config, model, tokenizer, loading, *, dependency_head=False):
    from huggingface_hub import hf_hub_download
    model_id, revision = config["model"]["pretrained_id"], config["model"]["revision"]
    if model.config._commit_hash != revision:
        raise ValueError("Loaded model revision differs from immutable request")
    if loading.get("missing_keys") or loading.get("unexpected_keys") or loading.get("error_msgs"):
        raise ValueError("Loaded base checkpoint has unexplained missing/unexpected parameters")
    mismatches = loading.get("mismatched_keys", [])
    # Transformers 4.57 reports key strings; older versions reported key/shape tuples.
    mismatch_names = [item if isinstance(item,str) else item[0] for item in mismatches]
    if (not dependency_head and mismatches) or any(name not in {"classifier.bias","classifier.weight"} for name in mismatch_names):
        raise ValueError("Unexpected checkpoint shape changes")
    weights = hf_hub_download(model_id,"model.safetensors",revision=revision,local_files_only=True)
    return {"model_id":model_id,"revision":revision,"base_weight_sha256":sha256_file(weights),
            "model_configuration":model.config.to_dict(),"tokenizer_class":type(tokenizer).__name__,
            "tokenizer_revision":revision,"loading_info":loading,
            "head": "new two-class dependency head, seeded initial fine-tuning" if dependency_head else "unchanged three-class NLI head"}


def validate_run_binding(config, manifest, resume_checkpoint=None):
    """Reject mutable weights and incompatible resume before expensive loading."""
    if not re.fullmatch(r"[0-9a-f]{40}", config["model"]["revision"]):
        raise ValueError("Model runs require an immutable revision")
    binding = {"model_id": config["model"]["pretrained_id"], "revision": config["model"]["revision"],
               "split_manifest_hash": sha256(manifest), "config_hash": sha256(config)}
    target = resolve_path(config, "checkpoints")
    if resume_checkpoint:
        checkpoint = Path(resume_checkpoint).resolve()
        required_files = ("trainer_state.json", "model.safetensors", "optimizer.pt", "scheduler.pt", "config.json")
        if not checkpoint.is_relative_to(target.resolve()) or not all((checkpoint/name).is_file() for name in required_files):
            raise ValueError("Resume requires a real checkpoint inside this run")
        recorded = json.loads((target / "run_binding.json").read_text(encoding="utf-8"))
        if recorded != binding:
            raise ValueError("Resume model, configuration or split binding differs")
    elif target.exists() and any(target.iterdir()):
        raise FileExistsError("Checkpoint directory is not empty; choose a new run or explicit resume checkpoint")
    return binding


def select_threshold(labels: list[int], probabilities: list[float], candidates: list[float]) -> float:
    if not labels or len(labels) != len(probabilities) or not candidates:
        raise ValueError("Validation labels/probabilities and threshold candidates required")
    calibration_metrics(labels, probabilities)
    if any(not 0 <= threshold <= 1 for threshold in candidates):
        raise ValueError("Thresholds must be in [0,1]")
    def score(threshold):
        metrics = binary_metrics(labels, [int(p >= threshold) for p in probabilities])
        return (metrics["f1"] or 0.0, metrics["recall"] or 0.0, -threshold)
    return max(candidates, key=score)


def _prediction_report(rows: list[dict], probabilities: list[float], threshold: float, bins: int,
                       raw_outputs: list[dict] | None = None) -> dict:
    if len(rows) != len(probabilities):
        raise ValueError("Predictions must align with every input row")
    calibration = calibration_metrics([row["label"] for row in rows], probabilities, bins)
    labels = [row["label"] for row in rows]
    predictions = [int(p >= threshold) for p in probabilities]
    positive = binary_metrics(labels, predictions)
    negative = binary_metrics([1-y for y in labels], [1-p for p in predictions])
    examples = [{**row, "probability": probability, "class_probabilities": [1-probability, probability],
                 "prediction": prediction, "correct": row["label"] == prediction,
                 "predicted_label_probability": probability if prediction else 1-probability}
                for row, probability, prediction in zip(rows, probabilities, predictions)]
    if raw_outputs is not None:
        if len(raw_outputs) != len(rows):
            raise ValueError("Raw model outputs must align with inputs")
        for example, raw in zip(examples, raw_outputs):
            example.update(raw)
    return {"threshold": threshold, "classification": positive,
            "class_counts": {"independent": labels.count(0), "depends_on": labels.count(1)},
            "confusion_matrix": [[sum(y == a and p == b for y,p in zip(labels,predictions))
                                  for b in (0,1)] for a in (0,1)],
            "matrix_order": ["independent", "depends_on"],
            "per_class": {"independent": {**negative,"support":labels.count(0)},
                          "depends_on": {**positive,"support":labels.count(1)}},
            "macro_f1": ((positive["f1"] or 0) + (negative["f1"] or 0))/2,
            "false_negative_dependencies": positive["fn"], "calibration": calibration,
            "calibration_interpretation": "Descriptive binned scores on this sample; not proof of calibration. NLI proxy scores have a different training objective.",
            "predictions": examples,
            "examples": {"correct": [x for x in examples if x["correct"]][:3],
                         "incorrect": [x for x in examples if not x["correct"]][:3],
                         "high_confidence_incorrect": [x for x in examples if not x["correct"] and x["predicted_label_probability"] >= .9]}}


def _infer(config: dict, rows: list[dict], model, tokenizer, positive_label: int,
           capture: list[dict] | None = None) -> list[float]:
    import torch

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    probabilities: list[float] = []
    batch_size = config["model"]["inference_batch_size"]
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Inference batch size must be positive")
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        tokens = tokenizer([row["premise"] for row in batch], [row["hypothesis"] for row in batch],
                           padding=True, truncation=True, max_length=config["model"]["max_length"],
                           return_tensors="pt")
        tokens = {name: tensor.to(device) for name, tensor in tokens.items()}
        with torch.inference_mode():
            logits = model(**tokens).logits
            complete = logits.softmax(dim=-1).cpu().tolist()
            probabilities.extend(row[positive_label] for row in complete)
            if capture is not None:
                capture.extend({"raw_logits":raw, "original_model_class_probabilities":{
                    model.config.id2label[index]:p for index,p in enumerate(probs)}}
                    for raw,probs in zip(logits.cpu().tolist(),complete))
    return probabilities


def run_pretrained_baseline(config: dict, *, manual: bool = False, include_test: bool = False,
                            selection_frozen: bool = False) -> dict:
    require_manual(config, "allow_model_inference", manual)
    if include_test and selection_frozen is not True:
        raise RuntimeError("Freeze model/threshold selection before held-out test evaluation")
    if any((resolve_path(config,"results")/name).exists() for name in ("pretrained_baseline.json","pretrained_selection.json")):
        raise FileExistsError("Pretrained records already exist; choose a new result directory")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

    splits, manifest = load_verified_splits(config)
    if not re.fullmatch(r"[0-9a-f]{40}", config["model"]["revision"]):
        raise ValueError("Pretrained inference requires an immutable revision")
    set_seed(config["training"]["seed"])
    model_id, revision = config["model"]["pretrained_id"], config["model"]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    model, loading = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision,
        trust_remote_code=False, use_safetensors=True, output_loading_info=True)
    provenance = _loaded_provenance(config, model, tokenizer, loading)
    validate_label_order(model.config.to_dict(),["contradiction","entailment","neutral"])
    mapping = {label.lower(): int(index) for index, label in model.config.id2label.items()}
    positive = mapping[config["model"]["entailment_label"]]
    validation_raw, test_raw = [], []
    validation = _infer(config, splits["validation"], model, tokenizer, positive, validation_raw)
    threshold = select_threshold([row["label"] for row in splits["validation"]], validation,
                                 config["model"]["threshold_candidates"])
    # Save validation selection before opening the held-out split for inference.
    write_json(resolve_path(config, "results") / "pretrained_selection.json",
               {"threshold": threshold, "model_id": model_id, "revision": revision,
                "split_manifest_hash": sha256(manifest), "config_hash": sha256(config)})
    test = _infer(config, splits["test"], model, tokenizer, positive, test_raw) if include_test else None
    report = {"status": "RUN_MANUALLY", "method": "pretrained NLI entailment proxy; not causal necessity",
              "model_id": model_id, "requested_revision": revision,
              "resolved_revision": getattr(model.config, "_commit_hash", None),
              "split_manifest_hash": sha256(manifest), "config_hash": sha256(config),
              "loaded_provenance": provenance,
              "validation": _prediction_report(splits["validation"], validation, threshold, config["model"]["calibration_bins"], validation_raw),
              "test": (_prediction_report(splits["test"], test, threshold, config["model"]["calibration_bins"], test_raw)
                       if test is not None else {"status": "NOT RUN; selection must be frozen"})}
    write_json(resolve_path(config, "results") / "pretrained_baseline.json", report)
    return report


def _trainer_metrics(prediction) -> dict:
    predictions = prediction.predictions.argmax(axis=-1).tolist()
    labels = prediction.label_ids.tolist()
    positive = binary_metrics(labels, predictions)
    negative = binary_metrics([1 - label for label in labels], [1 - label for label in predictions])
    return {"accuracy": positive["accuracy"] or 0.0,
            "macro_f1": ((positive["f1"] or 0.0) + (negative["f1"] or 0.0)) / 2}


def run_finetuning(config: dict, *, manual: bool = False, resume_checkpoint: str | None = None) -> dict:
    require_manual(config, "allow_training", manual)
    splits, manifest = load_verified_splits(config)
    binding = validate_run_binding(config, manifest, resume_checkpoint)
    if (resolve_path(config,"checkpoints")/"selection.json").exists():
        raise FileExistsError("A completed selection already exists; preserve it and choose a new run")
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)

    hp = config["training"]
    if set(row["label"] for row in splits["train"]) != {0, 1}:
        raise ValueError("Training split must contain both dependency labels")
    if hp["evaluation_strategy"] != "epoch" or hp["checkpoint_strategy"] != "epoch":
        raise ValueError("This foundation supports aligned epoch evaluation/checkpointing")
    if hp["best_model_metric"] not in {"macro_f1", "accuracy", "loss"}:
        raise ValueError("Unsupported selection metric")
    if hp["greater_is_better"] != (hp["best_model_metric"] != "loss"):
        raise ValueError("Best-model metric direction is inconsistent")
    set_seed(hp["seed"])
    model_id, revision = config["model"]["pretrained_id"], config["model"]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    # The 3-class NLI head is replaced by a 2-class dependency head; it must be trained.
    model, loading = AutoModelForSequenceClassification.from_pretrained(
        model_id, revision=revision, num_labels=2, id2label={0: "independent", 1: "depends_on"},
        label2id={"independent": 0, "depends_on": 1}, ignore_mismatched_sizes=True, trust_remote_code=False,
        use_safetensors=True, output_loading_info=True)
    provenance = _loaded_provenance(config, model, tokenizer, loading, dependency_head=True)
    provenance["fresh_head_initial_sha256"] = sha256({name: value.detach().cpu().tolist()
        for name, value in model.named_parameters() if name in {"classifier.weight", "classifier.bias"}})
    provenance["initialization_seed"] = hp["seed"]
    def tokenize(batch):
        return tokenizer(batch["premise"], batch["hypothesis"], truncation=True,
                         max_length=config["model"]["max_length"])
    encoded = {}
    tokenization_audit = {}
    for name in ("train", "validation"):
        lengths = [len(tokenizer(row["premise"],row["hypothesis"],truncation=False)["input_ids"]) for row in splits[name]]
        tokenization_audit[name] = {"max_pair_tokens":max(lengths),"truncated_rows":sum(n>config["model"]["max_length"] for n in lengths)}
        if tokenization_audit[name]["truncated_rows"]:
            raise ValueError("Pilot token limit truncates task context; review before training")
        dataset = Dataset.from_list(splits[name]).rename_column("label", "labels")
        encoded[name] = dataset.map(tokenize, batched=True,
                                    remove_columns=[column for column in dataset.column_names if column != "labels"])
    target = resolve_path(config, "checkpoints")
    if resume_checkpoint is None:
        write_json(target / "run_binding.json", binding)
    args = TrainingArguments(output_dir=str(target), seed=hp["seed"], data_seed=hp["seed"],
        per_device_train_batch_size=hp["batch_size"], per_device_eval_batch_size=hp["batch_size"],
        learning_rate=hp["learning_rate"], num_train_epochs=hp["epochs"], weight_decay=hp["weight_decay"],
        gradient_accumulation_steps=hp["gradient_accumulation_steps"],
        eval_strategy=hp["evaluation_strategy"], save_strategy=hp["checkpoint_strategy"],
        save_total_limit=hp["save_total_limit"], load_best_model_at_end=True,
        metric_for_best_model=hp["best_model_metric"], greater_is_better=hp["greater_is_better"],
        fp16=hp["fp16"], report_to="none", push_to_hub=False,
        logging_steps=1, logging_first_step=True, optim="adamw_torch",
        dataloader_pin_memory=False)
    trainer = Trainer(model=model, args=args, train_dataset=encoded["train"],
        eval_dataset=encoded["validation"], processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=_trainer_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=hp["early_stopping_patience"],
                                        early_stopping_threshold=hp["early_stopping_threshold"])])
    started = perf_counter()
    try:
        train_output = trainer.train(resume_from_checkpoint=resume_checkpoint)
    except Exception as error:
        write_json(target / "training_failure.json", {"status":"ERROR","type":type(error).__name__,
                   "message":str(error),"global_step":trainer.state.global_step,
                   "log_history":trainer.state.log_history,"elapsed_seconds":perf_counter()-started})
        raise
    trainer.save_model(str(target / "best"))
    tokenizer.save_pretrained(str(target / "best"))
    trainer.save_state()
    validation = _infer(config, splits["validation"], model, tokenizer, 1)
    threshold = select_threshold([row["label"] for row in splits["validation"]], validation,
                                 config["model"]["threshold_candidates"])
    selection = {"status": "TRAINED_MANUALLY", "selected_checkpoint": trainer.state.best_model_checkpoint,
                 "training_mode": "explicit compatible resume" if resume_checkpoint else "initial fine-tuning",
                 "resume_checkpoint": resume_checkpoint,
                 "validation_selection_metric": trainer.state.best_metric, "hyperparameters": hp,
                 "model_id": model_id, "requested_revision": revision,
                 "resolved_revision": getattr(model.config, "_commit_hash", None),
                 "split_manifest_hash": sha256(manifest), "config_hash": sha256(config),
                 "loaded_provenance": provenance, "tokenization_audit": tokenization_audit,
                 "threshold": threshold, "threshold_selection": "validation only, frozen before held-out evaluation",
                 "validation": _prediction_report(splits["validation"], validation, threshold, config["model"]["calibration_bins"]),
                 "training_metrics": train_output.metrics, "log_history": trainer.state.log_history,
                 "actual_optimizer_steps": trainer.state.global_step,
                 "checkpoint_file_hashes": {p.name: sha256_file(p) for p in (target / "best").iterdir() if p.is_file()},
                 "training_and_selection_seconds": perf_counter()-started,
                 "checkpoint_sha256": sha256_file(target / "best/model.safetensors"),
                 "test_evaluation": "NOT RUN by training; call run_model_evaluation once selection is frozen"}
    write_json(target / "selection.json", selection)
    return selection


def run_model_evaluation(config: dict, *, manual: bool = False, selection_frozen: bool = False) -> dict:
    require_manual(config, "allow_model_inference", manual)
    if selection_frozen is not True:
        raise RuntimeError("Freeze model/threshold selection before held-out test evaluation")
    if (resolve_path(config,"results")/"finetuned_evaluation.json").exists():
        raise FileExistsError("Held-out results already exist; use offline replay or a new result directory")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    splits, manifest = load_verified_splits(config)
    target = resolve_path(config, "checkpoints")
    selection = json.loads((target / "selection.json").read_text(encoding="utf-8"))
    if selection["split_manifest_hash"] != sha256(manifest):
        raise ValueError("Selected checkpoint belongs to different data splits")
    if selection["config_hash"] != sha256(config):
        raise ValueError("Selected checkpoint configuration changed after selection")
    if selection["checkpoint_sha256"] != sha256_file(target / "best/model.safetensors"):
        raise ValueError("Selected checkpoint bytes changed")
    model = AutoModelForSequenceClassification.from_pretrained(target / "best", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(target / "best", local_files_only=True)
    if model.config.id2label != {0: "independent", 1: "depends_on"}:
        raise ValueError("Checkpoint labels do not match the dependency task")
    threshold = selection["threshold"]
    test_raw = []
    test = _infer(config, splits["test"], model, tokenizer, 1, test_raw)
    report = {"status": "EVALUATED_MANUALLY", "selection": selection,
              "validation": selection["validation"],
              "test": _prediction_report(splits["test"], test, threshold, config["model"]["calibration_bins"], test_raw)}
    write_json(resolve_path(config, "results") / "finetuned_evaluation.json", report)
    return report
