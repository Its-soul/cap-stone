from __future__ import annotations

from copy import deepcopy
from pathlib import Path


def load_config(path: str | Path) -> dict:
    import yaml

    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    required = {"execution", "paths", "data", "model", "training", "recovery", "evaluation"}
    if not required.issubset(config):
        raise ValueError(f"Missing configuration sections: {required - set(config)}")
    for flag in ("allow_data_preparation", "allow_model_inference", "allow_training", "allow_research_experiments"):
        if type(config["execution"].get(flag)) is not bool:
            raise ValueError(f"Execution flag must be Boolean: {flag}")
    config = deepcopy(config)
    config["project_root"] = str(path.parent.parent)
    return config


def resolve_path(config: dict, name: str) -> Path:
    path = Path(config["paths"][name]).expanduser()
    return path if path.is_absolute() else Path(config["project_root"]) / path


def require_manual(config: dict, flag: str, manual: bool) -> None:
    if manual is not True or config["execution"].get(flag) is not True:
        raise RuntimeError("Prepared but intentionally not executed. Run manually in Google Colab. "
                           f"Enable {flag} and pass manual=True only for your chosen run.")

