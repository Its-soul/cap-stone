"""Offline preflight and synthetic direct-loader tests; no actual model executes."""

from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from cert_recovery.config import load_config
from cert_recovery.pinned_smoke import (prediction_from_logits, prepare_smoke, run_pinned_smoke,
                                       save_new, sha256_file, validate_label_order)

ROOT = Path(__file__).resolve().parents[1]


class PinnedSmokeTests(unittest.TestCase):
    def setUp(self):
        for method in ("connect", "connect_ex"):
            guard = patch("socket.socket." + method, side_effect=AssertionError("Offline tests prohibit network"))
            guard.start()
            self.addCleanup(guard.stop)
        self.config = load_config(ROOT / "config/config.yaml")

    def test_both_manual_guards_run_before_imports_or_downloads(self):
        with patch.dict("sys.modules", {"torch": None, "transformers": None, "huggingface_hub": None}):
            for manual in (False, True):
                with self.assertRaises(RuntimeError):
                    run_pinned_smoke(self.config, "deberta", manual=manual)
            self.config["execution"]["allow_model_inference"] = True
            with self.assertRaises(RuntimeError):
                run_pinned_smoke(self.config, "deberta")

    def test_preflight_binds_exact_pair_dataset_and_separate_models(self):
        first = prepare_smoke(ROOT, "deberta")
        second = prepare_smoke(ROOT, "minilm_fallback")
        self.assertEqual(first["cases"], second["cases"])
        self.assertEqual(first["cases"][0]["id"], "nli-003")
        self.assertEqual(first["cases"][0]["input"], {"premise": "A man is eating pizza.",
                                                     "hypothesis": "The man owns a bicycle."})
        self.assertEqual(first["cases"][0]["expected_label"], "neutral")
        self.assertEqual(first["dataset_sha256"], sha256_file(ROOT / "baseline_tests.json"))
        self.assertEqual(first["model"]["revision"], "fa2804872c3b4bd748f38c0185cc85775361e735")
        self.assertEqual(second["model"]["revision"], "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d")
        self.assertNotEqual(first["model"]["model_id"], second["model"]["model_id"])

    def temporary_project(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        (root / "config").mkdir()
        for relative in ("baseline_tests.json", "config/config.yaml", "config/pinned_smoke_models.json"):
            (root / relative).write_bytes((ROOT / relative).read_bytes())
        return root

    def test_mutable_pin_or_changed_dataset_is_rejected_before_loading(self):
        root = self.temporary_project()
        spec_path = root / "config/pinned_smoke_models.json"
        spec = json.loads(spec_path.read_text())
        spec["models"]["deberta"]["revision"] = "main"
        spec_path.write_text(json.dumps(spec))
        with self.assertRaisesRegex(ValueError, "immutable"):
            prepare_smoke(root, "deberta")
        spec_path.write_bytes((ROOT / "config/pinned_smoke_models.json").read_bytes())
        (root / "baseline_tests.json").write_bytes((root / "baseline_tests.json").read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            prepare_smoke(root, "deberta")

    def test_label_order_checks_configuration_instead_of_assuming_space_order(self):
        plan = prepare_smoke(ROOT, "deberta")
        config = deepcopy(plan["model"]["reviewed_config"])
        order = plan["model"]["reviewed_label_order"]
        self.assertEqual(validate_label_order(config, order), order)
        config["id2label"] = {int(key): value for key, value in config["id2label"].items()}
        self.assertEqual(validate_label_order(config, order), order)
        config["id2label"][0], config["id2label"][2] = config["id2label"][2], config["id2label"][0]
        with self.assertRaisesRegex(ValueError, "label order"):
            validate_label_order(config, order)

    def test_logits_keep_full_probabilities_and_reject_nonfinite_values(self):
        order = ["contradiction", "entailment", "neutral"]
        result = prediction_from_logits([3., -2., 1.], order)
        self.assertEqual(result["canonical_prediction"], "contradiction")
        self.assertEqual(result["predicted_class_index"], 0)
        self.assertEqual(result["raw_logits"], [3., -2., 1.])
        self.assertEqual(set(result["class_probabilities"]), set(order))
        self.assertAlmostEqual(sum(result["class_probabilities"].values()), 1.)
        self.assertTrue(result["advisory_only"])
        for logits in ([1., 2.], [1., float("nan"), 3.], [1., float("inf"), 3.]):
            with self.assertRaises(ValueError):
                prediction_from_logits(logits, order)

    def mock_loader(self, root, *, missing_keys=False):
        # Synthetic package/tensor fixtures; nothing here observes real weights or logits.
        spec_path = root / "config/pinned_smoke_models.json"
        spec = json.loads(spec_path.read_text())
        selected = spec["models"]["deberta"]
        config_path = root / "mock_config.json"
        config_path.write_text(json.dumps(selected["reviewed_config"]))
        selected["config_sha256"] = sha256_file(config_path)
        spec_path.write_text(json.dumps(spec))
        weight_path = root / "mock_model.safetensors"
        weight_path.write_bytes(b"SYNTHETIC OFFLINE WEIGHT FIXTURE, NOT A MODEL")
        actual_config = SimpleNamespace(_commit_hash=selected["revision"],
                                        to_dict=lambda: selected["reviewed_config"])
        model = Mock(config=actual_config)
        model.to.return_value = model
        model.return_value.logits.detach.return_value.cpu.return_value.tolist.return_value = [[3., -2., 1.]]
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                                version=SimpleNamespace(cuda=None), manual_seed=Mock(), no_grad=nullcontext)
        auto_config = Mock()
        auto_config.from_pretrained.return_value = actual_config
        auto_tokenizer = Mock()
        auto_tokenizer.from_pretrained.return_value.return_value = {"input_ids": Mock()}
        auto_model = Mock()

        def load(*args, **kwargs):
            prepared_files = list((root / "results").glob("*.prepared.json"))
            self.assertEqual(len(prepared_files), 1, "Preflight must be persisted before weight loading")
            prepared = json.loads(prepared_files[0].read_text())
            self.assertEqual(prepared["cases"], prepare_smoke(root, "deberta")["cases"])
            self.assertEqual(prepared["execution_status"], "PREPARED")
            self.assertEqual(kwargs["revision"], selected["revision"])
            self.assertFalse(kwargs["trust_remote_code"])
            return model, {"missing_keys": ["classifier.weight"] if missing_keys else []}

        auto_model.from_pretrained.side_effect = load
        hub = SimpleNamespace(hf_hub_download=Mock(side_effect=lambda repo, filename, **kwargs:
                                                 str(config_path if filename == "config.json" else weight_path)))
        modules = {"torch": torch, "huggingface_hub": hub,
                   "transformers": SimpleNamespace(AutoConfig=auto_config, AutoTokenizer=auto_tokenizer,
                                                   AutoModelForSequenceClassification=auto_model)}
        self.config["project_root"] = str(root)
        self.config["execution"]["allow_model_inference"] = True
        return modules, model, hub

    def test_synthetic_loader_records_preflight_logits_digest_and_mismatch(self):
        root = self.temporary_project()
        modules, model, hub = self.mock_loader(root)
        with patch.dict("sys.modules", modules), patch("cert_recovery.pinned_smoke.version", return_value="4.57.1"):
            result = run_pinned_smoke(self.config, "deberta", manual=True)
        self.assertEqual(result["execution_status"], "SUCCESS")
        self.assertEqual(result["loaded_weights"]["sha256"], sha256_file(root / "mock_model.safetensors"))
        self.assertFalse(result["predictions"][0]["matches_expected_label"])
        self.assertEqual(result["predictions"][0]["canonical_prediction"], "contradiction")
        self.assertEqual(result["workflow"], "direct_pinned_model_smoke_without_space_wrapper")
        self.assertEqual(len(list((root / "results").glob("*.json"))), 2)
        self.assertEqual(json.loads((root / "results" / (result["run_id"] + ".json")).read_text()), result)
        model.assert_called_once()
        for call in hub.hf_hub_download.call_args_list:
            self.assertEqual(call.kwargs["revision"], result["model"]["revision"])

    def test_missing_checkpoint_parameters_are_logged_and_never_inferred(self):
        root = self.temporary_project()
        modules, model, _ = self.mock_loader(root, missing_keys=True)
        with patch.dict("sys.modules", modules), patch("cert_recovery.pinned_smoke.version", return_value="4.57.1"):
            with self.assertRaisesRegex(ValueError, "load exactly"):
                run_pinned_smoke(self.config, "deberta", manual=True)
        model.assert_not_called()
        result = next(path for path in (root / "results").glob("*.json") if not path.name.endswith(".prepared.json"))
        saved = json.loads(result.read_text())
        self.assertEqual(saved["execution_status"], "ERROR")
        self.assertEqual(saved["predictions"], [])
        self.assertEqual(saved["error"]["type"], "ValueError")

    def test_smoke_serialization_never_overwrites_previous_files(self):
        root = self.temporary_project()
        path = root / "saved.json"
        save_new(path, {"preserve": True})
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            save_new(path, {"preserve": False})
        self.assertEqual(path.read_bytes(), before)
