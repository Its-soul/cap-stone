"""Author preparation notebooks; never execute their cells."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = "Prepared but intentionally not executed. Run manually in Google Colab."

SETUP = '''from pathlib import Path
import importlib.util
import subprocess
import sys
import zipfile

PROJECT_DIR = Path.cwd() / "certificate-recovery"
ARCHIVE = Path.cwd() / "Dependency_Certificate_Recovery_Foundation.zip"
if not PROJECT_DIR.exists():
    if not ARCHIVE.exists():
        from google.colab import files
        uploaded = files.upload()
        archives = [name for name in uploaded if name.endswith(".zip")]
        if len(archives) != 1:
            raise ValueError("Upload exactly one project ZIP")
        ARCHIVE = Path.cwd() / archives[0]
    with zipfile.ZipFile(ARCHIVE) as bundle:
        destination = Path.cwd().resolve()
        for member in bundle.infolist():
            if not (destination / member.filename).resolve().is_relative_to(destination):
                raise ValueError("Unsafe archive path")
        bundle.extractall(destination)
if not (PROJECT_DIR / "src" / "cert_recovery").exists():
    raise FileNotFoundError("Project source directory missing")
sys.path.insert(0, str(PROJECT_DIR / "src"))

INSTALL_CORE_PACKAGES = False
INSTALL_MODEL_PACKAGES = False
if INSTALL_CORE_PACKAGES:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(PROJECT_DIR / "requirements.txt")])
if INSTALL_MODEL_PACKAGES:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(PROJECT_DIR / "requirements-colab.txt")])
if importlib.util.find_spec("yaml") is None:
    raise RuntimeError("Set INSTALL_CORE_PACKAGES=True and rerun setup; no model packages are needed for the core.")

from cert_recovery.config import load_config
config = load_config(PROJECT_DIR / "config" / "config.yaml")
print("Prepared but intentionally not executed. Run manually in Google Colab.")
'''


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    compile(text, "prepared-cell", "exec")
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def write_notebook(filename, title, instructions, body):
    cells = [markdown(f"# {title}\n\n**NOT RUN**\n\n{STATUS}\n\n{instructions}\n\n"
                      "Upload the project ZIP when prompted. Set only the flags for the stage you intend to run. "
                      "Model stages need `INSTALL_MODEL_PACKAGES=True`; core stages need only PyYAML. "
                      "Back up datasets/checkpoints/results to Drive manually before the Colab session ends.\n"), code(SETUP)]
    cells.extend(body)
    for index, cell in enumerate(cells):
        cell["id"] = hashlib.sha256(f"{filename}:{index}".encode()).hexdigest()[:12]
    notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.11"},
                             "colab": {"name": filename}, "execution_status": "NOT RUN"}}
    (ROOT / "notebooks" / filename).write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    write_notebook("01_data_preparation.ipynb", "01 — Data preparation", "CPU only. Supply reviewed JSONL; no dataset download occurs.", [
        markdown("## Inputs\nThe format examples are small authored examples, not research data. "
                 "Use `USE_FORMAT_EXAMPLES` only for a manual setup check. Group related worlds/templates before splitting."),
        code('''import shutil
from cert_recovery.config import resolve_path

USE_FORMAT_EXAMPLES = False
if USE_FORMAT_EXAMPLES:
    for example, target_name in [("dependency_pairs.example.jsonl", "raw_pairs"), ("workflow_cases.example.jsonl", "workflow_cases")]:
        target = resolve_path(config, target_name)
        if target.exists():
            raise FileExistsError(f"Preserving existing input: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_DIR / "data/examples" / example, target)
'''),
        code('''from cert_recovery.data_pipeline import prepare_data

RUN_DATA_PREPARATION = False
if RUN_DATA_PREPARATION:
    config["execution"]["allow_data_preparation"] = True
    manifest = prepare_data(config, manual=True)
    print(manifest)
else:
    print("NOT RUN: data preparation")
'''),
        markdown("Inspect class balance, grouping and labels manually before model work. Preserve split_manifest.json; "
                 "do not repeatedly resplit based on test performance.")])

    write_notebook("02_baseline_model.ipynb", "02 — Pretrained semantic baseline", "Optional model inference. Run notebook 01 first and pin a model revision.", [
        markdown("The pretrained NLI entailment score is a limited dependency proxy. This stage uses validation only; "
                 "held-out test comparison belongs to notebook 04 after model selection is frozen."),
        code('''from cert_recovery.model_pipeline import run_pretrained_baseline

RUN_BASELINE = False
if RUN_BASELINE:
    config["execution"]["allow_model_inference"] = True
    baseline = run_pretrained_baseline(config, manual=True, include_test=False)
    print(baseline["validation"]["classification"])
else:
    print("NOT RUN: pretrained model inference")
''')])

    write_notebook("03_finetuning.ipynb", "03 — Semantic dependency fine-tuning", "Optional GPU training. Run notebook 01 first; confirm adequate Colab memory and reviewed labels.", [
        markdown("The binary head is newly initialized; it has no usable dependency performance before training. "
                 "All hyperparameters live in config.yaml. Use fresh checkpoint directories per run. "
                 "Early stopping and best-checkpoint selection use validation, never test."),
        code('''from cert_recovery.model_pipeline import run_finetuning

RUN_TRAINING = False
RESUME_CHECKPOINT = None
if RUN_TRAINING:
    config["execution"]["allow_training"] = True
    selection = run_finetuning(config, manual=True, resume_checkpoint=RESUME_CHECKPOINT)
    print(selection)
else:
    print("NOT RUN: model training/fine-tuning")
''')])

    write_notebook("04_model_evaluation.ipynb", "04 — Frozen model evaluation", "Experiments E08/E09. Requires notebook 03 checkpoints and fixed split hashes.", [
        markdown("Freeze training, model revision, threshold candidates and annotation decisions before setting "
                 "SELECTION_FROZEN. Test results must not guide further model selection. "
                 "This compares a pretrained NLI proxy to a trained dependency classifier; the tasks differ. "
                 "Neither probability is a safety certificate."),
        code('''from cert_recovery.model_pipeline import run_pretrained_baseline, run_model_evaluation

RUN_EVALUATION = False
SELECTION_FROZEN = False
if RUN_EVALUATION:
    if not SELECTION_FROZEN:
        raise RuntimeError("Freeze selection before opening held-out results")
    config["execution"]["allow_model_inference"] = True
    pretrained = run_pretrained_baseline(config, manual=True, include_test=True, selection_frozen=True)
    finetuned = run_model_evaluation(config, manual=True, selection_frozen=True)
    print({"pretrained_proxy": pretrained["test"]["classification"],
           "fine_tuned_dependency": finetuned["test"]["classification"]})
else:
    print("NOT RUN: model evaluation and comparison")
'''),
        markdown("Report false-negative dependency errors, source/domain slices and calibration. "
                 "Do not compare headline F1 without understanding annotation validity.")])

    write_notebook("05_dependency_experiments.ipynb", "05 — Dependency change preparation", "Experiment E01. CPU only; this is a research run, so it remains disabled.", [
        code('''from cert_recovery.config import resolve_path
from cert_recovery.data_pipeline import read_jsonl
from cert_recovery.evaluation import run_workflow_case

RUN_DEPENDENCY_EXPERIMENTS = False
if RUN_DEPENDENCY_EXPERIMENTS:
    config["execution"]["allow_research_experiments"] = True
    cases = read_jsonl(resolve_path(config, "workflow_cases"))
    dependency_records = [run_workflow_case(case, "C_invalidation_only", config, manual=True) for case in cases]
    print([{ "world_id": row["world_id"], "impact": row["impact"] } for row in dependency_records])
else:
    print("NOT RUN: dependency experiments")
'''),
        markdown("Changes are externally announced, not automatically discovered from text. Semantic edge "
                 "classification is prepared in notebooks 02–04; it is never used to remove mandatory graph edges.")])

    write_notebook("06_invalidation_experiments.ipynb", "06 — Invalidation and integrity", "Experiments E02/E03/E06. CPU only; no model invocation.", [
        code('''from cert_recovery.config import resolve_path
from cert_recovery.data_pipeline import read_jsonl
from cert_recovery.evaluation import run_direct_invalidation_experiment, run_workflow_case, run_integrity_experiment

RUN_INVALIDATION_EXPERIMENTS = False
if RUN_INVALIDATION_EXPERIMENTS:
    config["execution"]["allow_research_experiments"] = True
    direct = run_direct_invalidation_experiment(config, manual=True)
    cases = read_jsonl(resolve_path(config, "workflow_cases"))
    transitive = [run_workflow_case(case, "C_invalidation_only", config, manual=True) for case in cases]
    integrity = run_integrity_experiment(config, manual=True)
    print({"direct": direct, "transitive": transitive, "integrity": integrity})
else:
    print("NOT RUN: invalidation and integrity experiments")
'''),
        markdown("Direct means a graph edge to the certificate; snapshot ancestry is transitive. "
                 "Hold journal checkpoint length/head independently. If the attacker controls both, "
                 "hash-chain verification cannot authenticate history.")])

    write_notebook("07_recovery_experiments.ipynb", "07 — Selective recovery and replacement", "Experiments E04/E07. CPU only; optional budget is configured in config.yaml.", [
        code('''from cert_recovery.config import resolve_path
from cert_recovery.data_pipeline import read_jsonl
from cert_recovery.evaluation import run_workflow_case, run_replacement_experiment

RUN_RECOVERY_EXPERIMENTS = False
if RUN_RECOVERY_EXPERIMENTS:
    config["execution"]["allow_research_experiments"] = True
    cases = read_jsonl(resolve_path(config, "workflow_cases"))
    recovery_records = [run_workflow_case(case, "proposed", config, manual=True) for case in cases]
    replacement = run_replacement_experiment(config, manual=True)
    print({"recovery": recovery_records, "replacement": replacement})
else:
    print("NOT RUN: recovery experiments")
'''),
        markdown("Inspect blocked work, failed attempts and unaffected branches. Cheap abstention is not useful "
                 "recovery. A removed source needs a declared trusted replacement; recovery does not invent one.")])

    write_notebook("08_final_benchmark.ipynb", "08 — Future paired benchmark", "Experiments E05/E10. Foundation benchmark adapter; external validation remains future work.", [
        markdown("Freeze reviewed worlds and costs before running. The toy arithmetic adapter is for mechanism "
                 "inspection, not proof of general AI reliability. Baseline B is a deliberately stale cached "
                 "control; add strong literature baselines and external tasks before publication."),
        code('''from cert_recovery.evaluation import run_prepared_benchmark, run_fault_experiment, paired_cost_records

RUN_FINAL_BENCHMARK = False
STUDY_FROZEN = False
if RUN_FINAL_BENCHMARK:
    if not STUDY_FROZEN:
        raise RuntimeError("Freeze worlds, costs and policies before the final comparison")
    config["execution"]["allow_research_experiments"] = True
    benchmark = run_prepared_benchmark(config, manual=True)
    fault_records = run_fault_experiment(config, manual=True)
    print({"paired_cost": paired_cost_records(benchmark), "faults": fault_records})
else:
    print("NOT RUN: final benchmark and fault-injection experiments")
'''),
        markdown("Keep configured work units separate from measured latency. Compare useful completion and "
                 "support-validity at matched budgets; record failed-attempt/analysis overhead. "
                 "Repeat across independent worlds and report paired uncertainty before making research claims.")])


if __name__ == "__main__":
    main()

