import pytest
import os
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from collections import defaultdict
import subprocess
import sys

from cert_recovery.model_pipeline import _prediction_report
from cert_recovery.crypto import sha256

def test_duplicates_and_counterbalancing():
    # Verify our robust eval generator doesn't have duplicates and is counterbalanced
    from scripts.generate_robust_eval_data import generate_robust_suite
    pairs = generate_robust_suite()
    unique = set([(p['premise'], p['hypothesis']) for p in pairs])
    assert len(unique) == len(pairs) == 384

    stats = defaultdict(lambda: {'pos': 0, 'neg': 0})
    for p in pairs:
        src = p['derivation']['intervention']['source']
        if p['label'] == 1: stats[src]['pos'] += 1
        else: stats[src]['neg'] += 1

    for src, s in stats.items():
        assert s['pos'] == 48
        assert s['neg'] == 48

def test_metric_reconstruction():
    rows = [
        {"label": 1},
        {"label": 1},
        {"label": 0},
        {"label": 0}
    ]
    probs = [0.9, 0.4, 0.8, 0.1]
    # At threshold 0.5:
    # Preds: 1, 0, 1, 0
    # True : 1, 1, 0, 0
    # TP=1, FP=1, FN=1, TN=1

    rep = _prediction_report(rows, probs, 0.5, 10, [{}, {}, {}, {}])
    # The dictionary has 'per_class' which has support, tp, fn, etc. and 'macro_f1'
    # Actually wait, there is no global accuracy in the _prediction_report anymore.
    # It just returns per_class, macro_f1, etc.
    # Let's check macro_f1. True positive=1, FN=1, FP=1, TN=1.
    assert rep["macro_f1"] == .5
    assert rep["classification"]["accuracy"] == .5
    assert rep["confusion_matrix"] == [[1, 1], [1, 1]]
    assert [rep["per_class"][key]["support"] for key in ("independent", "depends_on")] == [2, 2]


def test_overwrite_prevention(tmp_path):
    out_dir = tmp_path / "eval_test"
    out_dir.mkdir()
    manifest = {"execution_status": "SUCCESS"}
    (out_dir / "manifest.json").write_text(json.dumps(manifest))

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_evaluation.py"

    result = subprocess.run(
        [sys.executable, str(script_path), "--checkpoint-run", str(tmp_path), "--eval-dataset", str(tmp_path), "--output-dir", str(out_dir)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, stdin=subprocess.DEVNULL
    )
    assert result.returncode != 0
    assert "Success result already exists" in result.stderr

def test_status_handling(tmp_path):
    out_dir = tmp_path / "eval_test2"

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_evaluation.py"
    # Provide bad paths to trigger failure
    result = subprocess.run(
        [sys.executable, str(script_path), "--checkpoint-run", str(tmp_path), "--eval-dataset", "missing.jsonl", "--output-dir", str(out_dir)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, stdin=subprocess.DEVNULL
    )
    assert result.returncode != 0
    # Check that manifest was saved as FAILED
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["execution_status"] == "FAILED"
    assert "error" in manifest


@pytest.mark.parametrize("status", ["FAILED", "PENDING", "ERROR", "TIMEOUT"])
def test_failed_attempts_are_preserved(tmp_path, status):
    out = tmp_path / "attempt"
    out.mkdir()
    original = json.dumps({"execution_status": status})
    (out / "manifest.json").write_text(original)
    script = Path(__file__).resolve().parents[1] / "scripts/run_evaluation.py"
    result = subprocess.run([sys.executable, str(script), "--checkpoint-run", str(tmp_path),
        "--eval-dataset", "missing.jsonl", "--output-dir", str(out)], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert result.returncode != 0 and "Attempt directory already exists" in result.stderr
    assert (out / "manifest.json").read_text() == original
