"""Read-only offline validation of an executed pilot bundle; never loads models."""
import argparse
import base64
import json
import math
from pathlib import Path
import re
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))
from cert_recovery.crypto import sha256
from cert_recovery.data_pipeline import load_verified_splits
from cert_recovery.pilot import WARNING, preserved_hashes
from cert_recovery.pinned_smoke import prepare_smoke, prediction_from_logits, sha256_file


def validate(run):
    run = Path(run).resolve()
    def reject_network(*args,**kwargs):
        raise AssertionError("Artifact validation prohibits network connections")
    socket.socket.connect = reject_network
    socket.socket.connect_ex = reject_network
    for path in run.rglob("*.json"):
        json.loads(path.read_text(encoding="utf-8-sig"),parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    config = json.loads((run/"config.json").read_text(encoding="utf-8"))
    manifest = json.loads((run/"manifest.json").read_text(encoding="utf-8"))
    assert manifest["preserved_hashes"] == preserved_hashes(Path(config["project_root"]))
    splits, split_manifest = load_verified_splits(config)
    assert [len(splits[k]) for k in ("train","validation","test")] == [64,16,16]
    assert all(sum(row["label"] for row in items)*2 == len(items) for items in splits.values())
    for path in (run/"metrics").glob("pinned_smoke_*.json"):
        if path.name.endswith(".prepared.json"):
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["execution_status"] != "SUCCESS":
            continue
        plan = prepare_smoke(ROOT,result["profile"],all_cases=True)
        assert [p["case"] for p in result["predictions"]] == plan["cases"]
        assert result["loaded_weights"]["revision"] == plan["model"]["revision"]
        assert result["resolved_config_revision"] == plan["model"]["revision"]
        assert len(result["loaded_weights"]["sha256"]) == 64
        for p in result["predictions"]:
            observed = prediction_from_logits(p["raw_logits"],plan["model"]["reviewed_label_order"])
            assert p["canonical_prediction"] == observed["canonical_prediction"]
            assert p["class_probabilities"] == observed["class_probabilities"]
            assert p["matches_expected_label"] == (p["canonical_prediction"] == p["case"]["expected_label"])
    for name in ("pretrained_baseline","finetuned_evaluation"):
        path = run/"metrics"/(name+".json")
        if not path.exists():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        test = result["test"]
        assert len(test["predictions"]) == len(splits["test"])
        assert {p["pair_id"] for p in test["predictions"]} == {p["pair_id"] for p in splits["test"]}
        assert sum(map(sum,test["confusion_matrix"])) == len(splits["test"])
        for p in test["predictions"]:
            assert all(math.isfinite(x) for x in p["raw_logits"])
            probs = p["original_model_class_probabilities"]
            assert all(0<=v<=1 for v in probs.values()) and math.isclose(sum(probs.values()),1,abs_tol=1e-6)
            original = next(row for row in splits["test"] if row["pair_id"] == p["pair_id"])
            assert all(p[k] == original[k] for k in original)
    selection_path = run/"checkpoints/selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        assert selection["split_manifest_hash"] == sha256(split_manifest)
        assert selection["config_hash"] == sha256(config)
        assert selection["checkpoint_sha256"] == sha256_file(run/"checkpoints/best/model.safetensors")
        assert any("loss" in item for item in selection["log_history"])
        assert any("eval_macro_f1" in item for item in selection["log_history"])
        evaluated = json.loads((run/"metrics/finetuned_evaluation.json").read_text(encoding="utf-8"))
        assert evaluated["test"]["threshold"] == selection["threshold"]
        assert evaluated["validation"] == selection["validation"]
    system_path = run/"metrics/workflow_benchmark.json"
    if system_path.exists():
        rows = json.loads(system_path.read_text(encoding="utf-8"))["records"]
        assert len(rows) == 144
        assert len({(r["world_id"],r["baseline"],r["repetition"]) for r in rows}) == 144
        assert all(r["safe_commits"] <= r["total_commits"] <= r["total_tasks"] for r in rows if r["status"] != "ERROR")
    page = (run/"report.html").read_text(encoding="utf-8")
    assert WARNING in page and "Execution status" in page
    encoded = re.findall(r"data:image/png;base64,([A-Za-z0-9+/=]+)",page)
    assert all(base64.b64decode(value).startswith(b"\x89PNG\r\n\x1a\n") for value in encoded)
    assert not re.search(r"(?:src|href)=['\"]https?://",page)
    for path in run.glob("*.png"):
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    print(f"Validated bundle: bound 64/16/16 splits, predictions/probabilities, selected checkpoint, 144 policy attempts, {len(encoded)} embedded plots, preserved history and standalone HTML.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    validate(parser.parse_args().run)
