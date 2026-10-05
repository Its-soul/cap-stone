"""Offline validation of public preparation; optionally check preserved research."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_NOTEBOOKS = ["02_baseline_model.ipynb"]
RESEARCH_NOTEBOOKS = [f"{index:02d}_{name}.ipynb" for index, name in enumerate(
    ["data_preparation", "baseline_model", "finetuning", "model_evaluation", "dependency_experiments",
     "invalidation_experiments", "recovery_experiments", "final_benchmark"], 1)]


def validate_notebooks(directory, expected):
    actual = sorted(path.name for path in directory.glob("*.ipynb"))
    assert actual == expected, f"Notebook names differ in {directory}: {actual}"
    code_cells = 0
    for filename in expected:
        notebook = json.loads((directory / filename).read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
        assert notebook["metadata"]["execution_status"] == "NOT RUN"
        ids = set()
        for cell in notebook["cells"]:
            assert cell["id"] not in ids
            ids.add(cell["id"])
            if cell["cell_type"] != "code":
                continue
            assert cell["execution_count"] is None and cell["outputs"] == []
            tree = ast.parse("".join(cell["source"]), filename=filename)
            for statement in tree.body:
                if isinstance(statement, ast.Assign):
                    for target in statement.targets:
                        if isinstance(target, ast.Name) and target.id.startswith(("RUN_", "INSTALL_", "USE_")):
                            assert isinstance(statement.value, ast.Constant) and statement.value.value is False, \
                                f"Execution flag must default to false: {filename}:{target.id}"
            code_cells += 1
    return code_cells


def validate_pins():
    smoke = json.loads((ROOT / "config/pinned_smoke_models.json").read_text(encoding="utf-8"))
    assert smoke["schema_version"] == 1 and smoke["status"] == "PREPARED_NOT_RUN"
    assert smoke["case_ids"] == ["nli-003"]
    assert smoke["dataset_sha256"] == hashlib.sha256((ROOT / "baseline_tests.json").read_bytes()).hexdigest()
    assert set(smoke["models"]) == {"deberta", "minilm_fallback"}
    for model in smoke["models"].values():
        revision = model["revision"]
        assert len(revision) == 40 and all(char in "0123456789abcdef" for char in revision)
        assert model["reviewed_label_order"] == ["contradiction", "entailment", "neutral"]
        assert model["reviewed_config"]["id2label"] == {
            str(index): label for index, label in enumerate(model["reviewed_label_order"])}
        assert model["reviewed_config"]["label2id"] == {
            label: index for index, label in enumerate(model["reviewed_label_order"])}
        assert f"/{revision}/" in model["config_url"] and f"/{revision}/" in model["model_card_url"]


def validate_documents():
    for path in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            target = target.split("#", 1)[0]
            resolved = (path.parent / target).resolve()
            assert resolved.is_relative_to(ROOT) and "_private" not in resolved.parts, \
                f"Public document depends on a private/external path: {path.name}"
            assert resolved.exists(), f"Broken public link: {path.name} -> {target}"


def validate_research(private_root):
    code_cells = validate_notebooks(private_root / "notebooks", RESEARCH_NOTEBOOKS)
    experiments = yaml.safe_load((private_root / "config/experiments.yaml").read_text(encoding="utf-8"))
    assert len(experiments["experiments"]) == 10
    assert {item["id"] for item in experiments["experiments"]} == {f"E{index:02d}" for index in range(1, 11)}
    for item in experiments["experiments"]:
        assert item["status"] == "NOT RUN" and item["results"] is None
        assert item["notebook"] in RESEARCH_NOTEBOOKS
        for field in ("objective", "input", "variables", "baseline", "method", "metrics", "expected_output"):
            assert item[field]
    status = json.loads((private_root / "results/EXPERIMENT_STATUS.json").read_text(encoding="utf-8"))
    assert status["results"] is None
    assert all(status[key] == "NOT RUN" for key in
               ("status", "model_training", "model_evaluation", "research_experiments"))
    for path in (private_root / "results").glob("*.json"):
        if path.name == "EXPERIMENT_STATUS.json":
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        assert path.name.startswith(("baseline_", "live_validation_"))
        assert record["schema_version"] in {1, 2} and record["run_id"] == path.stem
        assert record["remote"]["is_project_model"] is False
        if path.name.startswith("live_validation_"):
            assert record["schema_version"] == 2 and record["validation_type"] == "bounded_live_validation"
            budget = record["prediction_budget"]
            assert 0 <= budget["total_attempts"] <= budget["maximum"] <= 7
            assert budget["total_attempts"] == budget["baseline_attempts"] + budget["integration_attempts"]
    print(f"Private preparation: 8 preserved notebooks, {code_cells} code cells, 10 UNRUN experiments and historical log schemas.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", action="store_true", help="Also validate locally preserved research, if present")
    args = parser.parse_args()
    code_cells = validate_notebooks(ROOT / "notebooks", PUBLIC_NOTEBOOKS)
    validate_pins()
    validate_documents()
    config = yaml.safe_load((ROOT / "config/config.yaml").read_text(encoding="utf-8"))
    assert all(flag is False for flag in config["execution"].values())
    for path in (ROOT / "src").rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for relative in ("baseline_tests.json", ".env.example", "run_baseline.py", "remote_model_client.py",
                     "semantic_advice.py", "requirements-colab.txt", "tests/remote/fixtures/case_003_recorded_response.json"):
        assert (ROOT / relative).is_file(), f"Missing public workflow input: {relative}"
    print(f"Public preparation: 1 unexecuted notebook, {code_cells} code cells, 2 immutable profiles, disabled flags and valid documentation links.")
    if args.private:
        validate_research(ROOT / "_private/research")


if __name__ == "__main__":
    main()
