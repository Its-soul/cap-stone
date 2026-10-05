from __future__ import annotations

import json

from .config import require_manual, resolve_path
from .crypto import sha256
from .data_pipeline import load_verified_splits, write_json
from .metrics import binary_metrics, calibration_metrics


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


def _prediction_report(rows: list[dict], probabilities: list[float], threshold: float, bins: int) -> dict:
    labels = [row["label"] for row in rows]
    predictions = [int(p >= threshold) for p in probabilities]
    return {"threshold": threshold, "classification": binary_metrics(labels, predictions),
            "calibration": calibration_metrics(labels, probabilities, bins),
            "predictions": [{"pair_id": row["pair_id"], "probability": probability, "prediction": prediction}
                            for row, probability, prediction in zip(rows, probabilities, predictions)]}


def _infer(config: dict, rows: list[dict], model, tokenizer, positive_label: int) -> list[float]:
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
            probabilities.extend(model(**tokens).logits.softmax(dim=-1)[:, positive_label].cpu().tolist())
    return probabilities


def run_pretrained_baseline(config: dict, *, manual: bool = False, include_test: bool = False,
                            selection_frozen: bool = False) -> dict:
    require_manual(config, "allow_model_inference", manual)
    if include_test and selection_frozen is not True:
        raise RuntimeError("Freeze model/threshold selection before held-out test evaluation")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

    splits, manifest = load_verified_splits(config)
    set_seed(config["training"]["seed"])
    model_id, revision = config["model"]["pretrained_id"], config["model"]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    mapping = {label.lower(): int(index) for index, label in model.config.id2label.items()}
    positive = mapping[config["model"]["entailment_label"]]
    validation = _infer(config, splits["validation"], model, tokenizer, positive)
    threshold = select_threshold([row["label"] for row in splits["validation"]], validation,
                                 config["model"]["threshold_candidates"])
    test = _infer(config, splits["test"], model, tokenizer, positive) if include_test else None
    report = {"status": "RUN_MANUALLY", "method": "pretrained NLI entailment proxy; not causal necessity",
              "model_id": model_id, "requested_revision": revision,
              "resolved_revision": getattr(model.config, "_commit_hash", None),
              "split_manifest_hash": sha256(manifest), "config_hash": sha256(config),
              "validation": _prediction_report(splits["validation"], validation, threshold, config["model"]["calibration_bins"]),
              "test": (_prediction_report(splits["test"], test, threshold, config["model"]["calibration_bins"])
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
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)

    splits, manifest = load_verified_splits(config)
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
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, revision=revision, num_labels=2, id2label={0: "independent", 1: "depends_on"},
        label2id={"independent": 0, "depends_on": 1}, ignore_mismatched_sizes=True, trust_remote_code=False)
    def tokenize(batch):
        return tokenizer(batch["premise"], batch["hypothesis"], truncation=True,
                         max_length=config["model"]["max_length"])
    encoded = {}
    for name in ("train", "validation"):
        dataset = Dataset.from_list(splits[name]).rename_column("label", "labels")
        encoded[name] = dataset.map(tokenize, batched=True,
                                    remove_columns=[column for column in dataset.column_names if column != "labels"])
    target = resolve_path(config, "checkpoints")
    if target.exists() and any(target.iterdir()) and resume_checkpoint is None:
        raise FileExistsError("Checkpoint directory is not empty; choose a new run or explicit resume checkpoint")
    args = TrainingArguments(output_dir=str(target), seed=hp["seed"], data_seed=hp["seed"],
        per_device_train_batch_size=hp["batch_size"], per_device_eval_batch_size=hp["batch_size"],
        learning_rate=hp["learning_rate"], num_train_epochs=hp["epochs"], weight_decay=hp["weight_decay"],
        gradient_accumulation_steps=hp["gradient_accumulation_steps"],
        eval_strategy=hp["evaluation_strategy"], save_strategy=hp["checkpoint_strategy"],
        save_total_limit=hp["save_total_limit"], load_best_model_at_end=True,
        metric_for_best_model=hp["best_model_metric"], greater_is_better=hp["greater_is_better"],
        fp16=hp["fp16"], report_to="none", push_to_hub=False)
    trainer = Trainer(model=model, args=args, train_dataset=encoded["train"],
        eval_dataset=encoded["validation"], processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=_trainer_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=hp["early_stopping_patience"],
                                        early_stopping_threshold=hp["early_stopping_threshold"])])
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(str(target / "best"))
    tokenizer.save_pretrained(str(target / "best"))
    selection = {"status": "TRAINED_MANUALLY", "selected_checkpoint": trainer.state.best_model_checkpoint,
                 "validation_selection_metric": trainer.state.best_metric, "hyperparameters": hp,
                 "model_id": model_id, "requested_revision": revision,
                 "resolved_revision": getattr(model.config, "_commit_hash", None),
                 "split_manifest_hash": sha256(manifest), "config_hash": sha256(config),
                 "test_evaluation": "NOT RUN by training; call run_model_evaluation once selection is frozen"}
    write_json(target / "selection.json", selection)
    return selection


def run_model_evaluation(config: dict, *, manual: bool = False, selection_frozen: bool = False) -> dict:
    require_manual(config, "allow_model_inference", manual)
    if selection_frozen is not True:
        raise RuntimeError("Freeze model/threshold selection before held-out test evaluation")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    splits, manifest = load_verified_splits(config)
    target = resolve_path(config, "checkpoints")
    selection = json.loads((target / "selection.json").read_text(encoding="utf-8"))
    if selection["split_manifest_hash"] != sha256(manifest):
        raise ValueError("Selected checkpoint belongs to different data splits")
    model = AutoModelForSequenceClassification.from_pretrained(target / "best", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(target / "best", local_files_only=True)
    if model.config.id2label != {0: "independent", 1: "depends_on"}:
        raise ValueError("Checkpoint labels do not match the dependency task")
    validation = _infer(config, splits["validation"], model, tokenizer, 1)
    threshold = select_threshold([row["label"] for row in splits["validation"]], validation,
                                 config["model"]["threshold_candidates"])
    test = _infer(config, splits["test"], model, tokenizer, 1)
    report = {"status": "EVALUATED_MANUALLY", "selection": selection,
              "validation": _prediction_report(splits["validation"], validation, threshold, config["model"]["calibration_bins"]),
              "test": _prediction_report(splits["test"], test, threshold, config["model"]["calibration_bins"])}
    write_json(resolve_path(config, "results") / "finetuned_evaluation.json", report)
    return report
