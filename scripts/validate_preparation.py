"""Validate artifact structure and syntax without executing notebook/workload code."""

import ast
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = [f"{index:02d}_{name}.ipynb" for index, name in enumerate(
    ["data_preparation", "baseline_model", "finetuning", "model_evaluation", "dependency_experiments",
     "invalidation_experiments", "recovery_experiments", "final_benchmark"], 1)]


def main():
    code_cells = 0
    actual = sorted(path.name for path in (ROOT / "notebooks").glob("*.ipynb"))
    assert actual == EXPECTED, f"Notebook names differ: {actual}"
    for filename in EXPECTED:
        notebook = json.loads((ROOT / "notebooks" / filename).read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
        assert notebook["metadata"]["execution_status"] == "NOT RUN"
        ids = set()
        for cell in notebook["cells"]:
            assert cell["id"] not in ids
            ids.add(cell["id"])
            source = "".join(cell["source"])
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None and cell["outputs"] == []
                ast.parse(source, filename=filename)
                code_cells += 1
    config = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    assert all(flag is False for flag in config["execution"].values())
    experiments = yaml.safe_load((ROOT / "config/experiments.yaml").read_text())
    assert len(experiments["experiments"]) == 10
    assert {item["id"] for item in experiments["experiments"]} == {f"E{index:02d}" for index in range(1, 11)}
    for item in experiments["experiments"]:
        assert item["status"] == "NOT RUN" and item["results"] is None
        assert item["notebook"] in EXPECTED
        for field in ("objective", "input", "variables", "baseline", "method", "metrics", "expected_output"):
            assert item[field]
    for path in (ROOT / "src").rglob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
    results = sorted(path.name for path in (ROOT / "results").iterdir())
    assert results == [".gitkeep", "EXPERIMENT_STATUS.json"], f"Unexpected generated results: {results}"
    print(f"Validated {len(EXPECTED)} unexecuted notebooks, {code_cells} code-cell syntaxes and 10 UNRUN experiments.")


if __name__ == "__main__":
    main()

