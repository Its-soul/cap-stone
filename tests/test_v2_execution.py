"""Frozen V2 artifacts and stage lifecycle, with external training mocked."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cert_recovery.config import load_config
from cert_recovery.controlled_study import execute_stage, verify_frozen
from cert_recovery.crypto import sha256
from cert_recovery.data_pipeline import prepare_data, write_json
from cert_recovery.pinned_smoke import sha256_file
import cert_recovery.controlled_study as study

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def frozen_run(tmp_path):
    """Six synthetic rows and real source hashes; no weights or private run data."""
    run = tmp_path / "run"
    run.mkdir()
    config = load_config(ROOT / "config/config.yaml")
    config["project_root"] = str(tmp_path)
    config["execution"].update(allow_data_preparation=True, allow_training=True, allow_model_inference=True)
    config["model"]["revision"] = "a" * 40  # Synthetic revision; never loaded.
    config["paths"].update(raw_pairs=str(run / "dependency_pairs.jsonl"),
                           processed=str(run / "splits"), checkpoints=str(run / "checkpoints"),
                           results=str(run / "metrics"))
    rows = []
    for split in ("train", "validation", "test"):
        for label in (0, 1):
            rows.append({"pair_id": f"{split}-{label}", "world_id": split,
                         "premise": f"Synthetic {split} source {label} is 17.",
                         "hypothesis": "Explicit synthetic support contract.", "label": label,
                         "source": "offline V2 fixture"})
    (run / "dependency_pairs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    split_manifest = prepare_data(config, manual=True,
                                  group_assignments={name: name for name in ("train", "validation", "test")})
    source = ROOT / "src/cert_recovery/model_pipeline.py"
    snapshot = run / "source_snapshot/model_pipeline.py"
    actual = tmp_path / "src/cert_recovery/model_pipeline.py"
    for target in (snapshot, actual):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    protocol = {"config_sha256": sha256(config), "split_manifest_sha256": sha256(split_manifest),
                "dataset_file_sha256": sha256_file(run / "dependency_pairs.jsonl"),
                "split_file_sha256": {name: sha256_file(run / "splits" / f"{name}.jsonl")
                                      for name in ("train", "validation", "test")},
                "source_files_sha256": {"model_pipeline.py": sha256_file(snapshot)},
                "budget": {"torch_threads": 4}}
    write_json(run / "config.json", config)
    write_json(run / "protocol.json", protocol)
    write_json(run / "manifest.json", {"protocol_sha256": sha256(protocol), "status": "PREPARED", "stages": {}})
    return run


def read(path):
    return json.loads(path.read_text())


def test_v2_verified_bundle_matches_its_frozen_configuration(frozen_run):
    config, protocol = verify_frozen(frozen_run)
    assert config == read(frozen_run / "config.json")
    assert protocol == read(frozen_run / "protocol.json")


@pytest.mark.parametrize("artifact,expected", [
    ("protocol.json", "Frozen protocol changed"),
    ("config.json", "Frozen configuration changed"),
    ("dependency_pairs.jsonl", "Frozen data changed"),
    ("splits/validation.jsonl", "altered split"),
    ("splits/split_manifest.json", "Frozen data changed"),
    ("source_snapshot/model_pipeline.py", "Source snapshot changed"),
    ("current_source", "Current implementation differs"),
])
def test_v2_rejects_tampered_frozen_artifacts(frozen_run, artifact, expected):
    if artifact == "current_source":
        path = frozen_run.parent / "src/cert_recovery/model_pipeline.py"
    else:
        path = frozen_run / artifact
    if path.name in ("protocol.json", "config.json", "split_manifest.json"):
        value = read(path)
        value["synthetic_tampering"] = True
        write_json(path, value)
    elif path.name == "validation.jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["premise"] = "Altered fixture text"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    else:
        with path.open("a") as stream:
            stream.write("\n")
    with pytest.raises(ValueError, match=expected):
        verify_frozen(frozen_run)


@pytest.fixture
def stage_backend(monkeypatch):
    torch = SimpleNamespace(set_num_threads=Mock(), cuda=SimpleNamespace(is_available=Mock(return_value=False)))
    monkeypatch.setitem(sys.modules, "torch", torch)
    callbacks = {}
    for name in ("run_finetuning", "run_pretrained_baseline", "compare"):
        callbacks[name] = Mock()
        monkeypatch.setattr(study, name, callbacks[name])
    return torch, callbacks


@pytest.mark.parametrize("stage", ["proxy_selection", "comparison"])
def test_v2_requires_successful_training_before_later_stages(frozen_run, stage_backend, stage):
    before = (frozen_run / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="Successful selected training required"):
        execute_stage(frozen_run, stage)
    assert (frozen_run / "manifest.json").read_bytes() == before
    assert not list(frozen_run.glob("attempt_*"))
    assert all(not callback.called for callback in stage_backend[1].values())


def test_v2_comparison_requires_frozen_proxy_selection(frozen_run, stage_backend):
    manifest = read(frozen_run / "manifest.json")
    manifest["stages"]["training"] = {"status": "SUCCESS"}
    write_json(frozen_run / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Freeze proxy selection"):
        execute_stage(frozen_run, "comparison")
    assert not list(frozen_run.glob("attempt_*"))
    stage_backend[1]["compare"].assert_not_called()


@pytest.mark.parametrize("stage,callback", [
    ("training", "run_finetuning"), ("proxy_selection", "run_pretrained_baseline"), ("comparison", "compare"),
])
def test_v2_success_records_stage_attempt_and_disallows_retry(frozen_run, stage_backend, stage, callback):
    manifest = read(frozen_run / "manifest.json")
    if stage != "training":
        manifest["stages"]["training"] = {"status": "SUCCESS"}
    if stage == "comparison":
        manifest["stages"]["proxy_selection"] = {"status": "SUCCESS"}
    write_json(frozen_run / "manifest.json", manifest)
    state = execute_stage(frozen_run, stage)
    assert state["status"] == "SUCCESS"
    assert state["elapsed_seconds"] >= 0
    assert state["started_utc"] and state["finished_utc"]
    assert read(Path(state["attempt"]) / "status.json") == state
    assert read(frozen_run / "manifest.json")["stages"][stage] == state
    stage_backend[0].set_num_threads.assert_called_once_with(4)
    selected = stage_backend[1][callback]
    selected.assert_called_once()
    if stage == "training":
        assert selected.call_args.kwargs == {"manual": True}
    elif stage == "proxy_selection":
        assert selected.call_args.kwargs == {"manual": True, "include_test": False}
    previous = (Path(state["attempt"]) / "status.json").read_bytes()
    with pytest.raises(FileExistsError, match="Stage already attempted"):
        execute_stage(frozen_run, stage)
    assert (Path(state["attempt"]) / "status.json").read_bytes() == previous
    assert len(list(frozen_run.glob("attempt_*"))) == 1
    selected.assert_called_once()


@pytest.mark.parametrize("error", [RuntimeError("synthetic training failure"), KeyboardInterrupt("synthetic interruption")])
def test_v2_failed_or_interrupted_stage_preserves_evidence_and_cannot_retry(frozen_run, stage_backend, error):
    stage_backend[1]["run_finetuning"].side_effect = error
    with pytest.raises(type(error)):
        execute_stage(frozen_run, "training")
    manifest = read(frozen_run / "manifest.json")
    state = manifest["stages"]["training"]
    assert manifest["status"] == state["status"] == "ERROR"
    assert state["error"] == {"type": type(error).__name__, "message": str(error)}
    assert read(Path(state["attempt"]) / "status.json") == state
    previous = deepcopy(manifest)
    with pytest.raises(FileExistsError):
        execute_stage(frozen_run, "training")
    assert read(frozen_run / "manifest.json") == previous
    stage_backend[1]["run_finetuning"].assert_called_once()


def test_v2_cpu_only_guard_records_failure_without_training(frozen_run, stage_backend):
    stage_backend[0].cuda.is_available.return_value = True
    with pytest.raises(ValueError, match="local CPU only"):
        execute_stage(frozen_run, "training")
    state = read(frozen_run / "manifest.json")["stages"]["training"]
    assert state["status"] == "ERROR"
    assert read(Path(state["attempt"]) / "status.json") == state
    stage_backend[1]["run_finetuning"].assert_not_called()
