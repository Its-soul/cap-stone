"""Author the disabled public pinned-model usage notebook without executing it."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILENAME = "02_baseline_model.ipynb"


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    compile(text, "prepared-cell", "exec")
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def public_notebook():
    cells = [markdown("# Optional pinned NLI smoke\n\n**NOT RUN**\n\n"
        "Use only in a separately authorized free Colab session. Place the project at "
        "the current directory or in certificate-recovery/. All installation/inference "
        "flags default to false. Keep model packages in a separate environment from "
        "the remote-client extra. See docs/MODEL.md for provenance and limits."),
        code('''from pathlib import Path
import subprocess
import sys

PROJECT_DIR = Path.cwd()
if not (PROJECT_DIR / "src/cert_recovery").exists():
    PROJECT_DIR = PROJECT_DIR / "certificate-recovery"
if not (PROJECT_DIR / "src/cert_recovery").exists():
    raise FileNotFoundError("Place the project in the current directory or certificate-recovery/")
sys.path.insert(0, str(PROJECT_DIR / "src"))

INSTALL_MODEL_PACKAGES = False
if INSTALL_MODEL_PACKAGES:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r",
                           str(PROJECT_DIR / "requirements-colab.txt")])

from cert_recovery.config import load_config
from cert_recovery.pinned_smoke import run_pinned_smoke
config = load_config(PROJECT_DIR / "config/config.yaml")
'''),
        markdown("Each comparison runs all six unchanged smoke pairs, including nli-003, premise first, "
                 "without a Space heuristic wrapper or research evaluation. Immutable "
                 "pins/configurations are in config/pinned_smoke_models.json. Fresh "
                 "results/pinned_smoke_*.prepared.json and a separate result capture "
                 "input/config/runtime provenance, weight digest, logits, full class "
                 "probabilities or errors. No inference retries or overwrites. Preserve "
                 "both files before the session ends. This cannot identify historical "
                 "Space weights; confidence has no certificate authority."),
        code('''RUN_DEBERTA_SMOKE = False
if RUN_DEBERTA_SMOKE:
    config["execution"]["allow_model_inference"] = True
    result = run_pinned_smoke(config, "deberta", manual=True, all_cases=True)
    print(result["run_id"], result["predictions"])
else:
    print("NOT RUN: pinned DeBERTa smoke")
'''),
        code('''RUN_MINILM_SMOKE = False
if RUN_MINILM_SMOKE:
    config["execution"]["allow_model_inference"] = True
    result = run_pinned_smoke(config, "minilm_fallback", manual=True, all_cases=True)
    print(result["run_id"], result["predictions"])
else:
    print("NOT RUN: separate pinned MiniLM fallback comparison")
'''), markdown("## Optional bounded synthetic pilot\n\n"
        "**PRELIMINARY / EXPERIMENTAL — not final or production results.**\n\n"
        "This runs the existing data/model/system pipelines: 96 source-sum pairs, "
        "12 isolated template families, 64/16/16 train/validation/test rows, two "
        "epochs on 64 training rows, immutable DeBERTa base weights, held-out "
        "selection and separate six-pair direct NLI comparisons. It never trains "
        "on the smoke pairs. Results, logs and checkpoints stay in a new ignored "
        "_private/runs/ directory. Prefer a free Colab GPU; no paid job is launched. "
        "Each model stage has a 30-minute timeout and is attempted once. Download "
        "the entire result directory before Colab ends. Opening report.html requires "
        "no server. The original eight private research notebooks remain unchanged."),
        code('''RUN_SYNTHETIC_PILOT = False
if RUN_SYNTHETIC_PILOT:
    subprocess.check_call([sys.executable, str(PROJECT_DIR / "scripts/run_pilot.py"), "--execute"])
else:
    print("NOT RUN: optional bounded synthetic pilot")
''')]
    for index, cell in enumerate(cells):
        cell["id"] = hashlib.sha256(f"{FILENAME}:{index}".encode()).hexdigest()[:12]
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"},
                         "execution_status": "NOT RUN"}}


def main():
    target = ROOT / "notebooks" / FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if any(cell.get("outputs") or cell.get("execution_count") is not None
               for cell in existing["cells"] if cell["cell_type"] == "code"):
            raise RuntimeError("Preserve executed notebook outputs before regeneration")
    target.write_text(json.dumps(public_notebook(), indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
