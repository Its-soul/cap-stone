"""Deterministic synthetic fixtures only; every network connection is blocked."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cert_recovery.config import load_config
from cert_recovery.crypto import sha256
from cert_recovery.data_pipeline import clean_rows, grouped_split, normalize_text, write_json
from cert_recovery.evaluation import run_workflow_case
from cert_recovery.model_pipeline import _prediction_report, validate_run_binding
from cert_recovery.model_pipeline import _loaded_provenance
from cert_recovery.pilot import synthetic_pairs, workflow_cases, WARNING, audit_synthetic_identifiers
from cert_recovery.pinned_smoke import prepare_smoke

ROOT = Path(__file__).resolve().parents[1]


class PilotTests(unittest.TestCase):
    def setUp(self):
        for method in ("connect", "connect_ex"):
            guard = patch("socket.socket."+method, side_effect=AssertionError("Offline fixtures prohibit network"))
            guard.start()
            self.addCleanup(guard.stop)
        self.config = load_config(ROOT / "config/config.yaml")

    def test_synthetic_labels_have_independent_declared_rule_and_intervention_witness(self):
        rows = synthetic_pairs(self.config["recovery"])
        self.assertEqual(len(rows),96)
        self.assertEqual(sum(r["label"] for r in rows),48)
        self.assertEqual(rows, synthetic_pairs(self.config["recovery"]))
        audit = audit_synthetic_identifiers(rows)
        self.assertEqual(audit["single_class_source_ids"],["D2"])
        self.assertEqual(audit["source_label_counts"]["D2"],[27,0])
        for row in rows:
            derivation = row["derivation"]
            _, source, claim = row["pair_id"].rsplit("-",2)
            source_index, claim_index = int(source[1:]), int(claim[1:])
            oracle = source_index <= claim_index if derivation["topology"] == "cascade" else source_index == claim_index
            self.assertEqual(row["label"],int(oracle))
            self.assertEqual(derivation["intervention"]["after"]-derivation["intervention"]["before"],row["label"])

    def test_family_split_keeps_related_worlds_and_templates_disjoint_and_balanced(self):
        raw = synthetic_pairs(self.config["recovery"])
        rows, cleaning = clean_rows(raw,"template_family")
        self.assertEqual(cleaning["removed_duplicates"],0)
        splits = grouped_split(rows,(.7,.15,.15),42,"template_family")
        self.assertEqual([len(splits[k]) for k in ("train","validation","test")],[64,16,16])
        assignments = {}
        for name, items in splits.items():
            self.assertEqual({r["label"] for r in items},{0,1})
            for row in items:
                self.assertIn(assignments.setdefault(row["template_family"],name),(name,))
        smoke = {(c["input"]["premise"],c["input"]["hypothesis"]) for c in prepare_smoke(ROOT,"deberta",all_cases=True)["cases"]}
        self.assertFalse(smoke & {(r["premise"],r["hypothesis"]) for r in rows})

    def test_cleaning_retains_numbers_and_negation(self):
        self.assertEqual(normalize_text("  D0 is not  -13.\nNo  other sources count. "),"D0 is not -13. No other sources count.")

    def test_both_smokes_keep_all_six_exact_directional_pairs(self):
        original = json.loads((ROOT / "baseline_tests.json").read_text())["cases"]
        for profile in ("deberta","minilm_fallback"):
            self.assertEqual(prepare_smoke(ROOT,profile,all_cases=True)["cases"],original)
            self.assertEqual(len(prepare_smoke(ROOT,profile)["cases"]),1)

    def test_confusion_per_class_and_high_confidence_errors_are_faithful(self):
        rows = [{"pair_id":str(i),"label":y} for i,y in enumerate([0,0,1,1])]
        report = _prediction_report(rows,[.01,.99,.01,.99],.5,10)
        self.assertEqual(report["confusion_matrix"],[[1,1],[1,1]])
        self.assertEqual(report["macro_f1"],.5)
        self.assertEqual(report["false_negative_dependencies"],1)
        self.assertEqual(report["per_class"]["depends_on"]["support"],2)
        self.assertEqual(len(report["examples"]["high_confidence_incorrect"]),2)
        self.assertAlmostEqual(report["calibration"]["brier"],.4901)
        with self.assertRaises(ValueError):
            _prediction_report(rows,[.5],.5,10)

    def test_loading_info_accepts_only_expected_binary_head_keys_in_457_format(self):
        from types import SimpleNamespace
        config = deepcopy(self.config)
        config["model"]["revision"] = "a"*40
        model = SimpleNamespace(config=SimpleNamespace(_commit_hash="a"*40,to_dict=lambda:{}))
        with tempfile.TemporaryDirectory() as temp:
            weights = Path(temp)/"model.safetensors"
            weights.write_bytes(b"SYNTHETIC LOADER FIXTURE, NOT WEIGHTS")
            with patch.dict("sys.modules",{"huggingface_hub":SimpleNamespace(hf_hub_download=lambda *a,**kw:str(weights))}):
                for mismatch in (["classifier.bias","classifier.weight"],[("classifier.bias",[3],[2])]):
                    result = _loaded_provenance(config,model,object(),{"mismatched_keys":mismatch},dependency_head=True)
                    self.assertIn("new two-class",result["head"])
                for loading in ({"mismatched_keys":["encoder.weight"]},{"missing_keys":["encoder.weight"]}):
                    with self.assertRaises(ValueError):
                        _loaded_provenance(config,model,object(),loading,dependency_head=True)
                with self.assertRaises(ValueError):
                    _loaded_provenance(config,model,object(),{"mismatched_keys":["classifier.bias"]})

    def test_resume_rejects_mutable_missing_or_incompatible_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            config = deepcopy(self.config)
            config["paths"]["checkpoints"] = str(Path(temp)/"checkpoints")
            with self.assertRaisesRegex(ValueError,"immutable"):
                validate_run_binding(config,{})
            config["model"]["revision"] = prepare_smoke(ROOT,"deberta")["model"]["revision"]
            manifest = {"fixture":"split"}
            binding = validate_run_binding(config,manifest)
            checkpoint = Path(config["paths"]["checkpoints"])/"checkpoint-1"
            checkpoint.mkdir(parents=True)
            write_json(checkpoint.parent / "run_binding.json",binding)
            with self.assertRaisesRegex(ValueError,"real checkpoint"):
                validate_run_binding(config,manifest,str(checkpoint))
            write_json(checkpoint / "trainer_state.json",{"global_step":1})
            for name in ("model.safetensors","optimizer.pt","scheduler.pt","config.json"):
                (checkpoint/name).write_bytes(b"SYNTHETIC FILE PRESENCE FIXTURE; NOT A REAL CHECKPOINT")
            self.assertEqual(validate_run_binding(config,manifest,str(checkpoint)),binding)
            with self.assertRaisesRegex(ValueError,"binding differs"):
                validate_run_binding(config,{"fixture":"altered"},str(checkpoint))
            with self.assertRaises(FileExistsError):
                validate_run_binding(config,manifest)

    def test_policy_measurements_count_unsafe_commits_blocking_and_setup(self):
        config = deepcopy(self.config)
        config["execution"]["allow_research_experiments"] = True
        case = workflow_cases()[0]
        observed = {policy: run_workflow_case(case,policy,config,manual=True)
                    for policy in ("A_full_recomputation","B_no_invalidation","C_invalidation_only","proposed")}
        self.assertEqual(observed["A_full_recomputation"]["safe_commits"],3)
        self.assertEqual(observed["proposed"]["safe_commits"],3)
        self.assertEqual(observed["B_no_invalidation"]["safe_commits"],2)
        self.assertEqual(observed["B_no_invalidation"]["total_commits"],3)
        self.assertEqual(observed["C_invalidation_only"]["blocked_tasks"],1)
        for row in observed.values():
            self.assertGreaterEqual(row["total_elapsed_seconds"],row["latency_seconds"])
            self.assertGreater(row["initial_setup_seconds"],0)
            self.assertGreater(row["audit_events"],0)

    def test_report_missing_stages_is_explicit_and_contains_no_dummy_metrics(self):
        from cert_recovery.pilot import finalize
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            write_json(run/"config.json",{"project_root":str(ROOT)})
            from cert_recovery.pilot import preserved_hashes
            write_json(run/"manifest.json",{"stages":{"training":{"status":"NOT RUN"}},
                       "preserved_hashes":preserved_hashes(ROOT),"limitations":["fixture; no execution"]})
            summary = finalize(run)
            self.assertEqual(summary["models"],{})
            page = (run/"report.html").read_text(encoding="utf-8")
            self.assertIn(WARNING,page)
            self.assertIn("NOT RUN",page)
            self.assertNotIn("data:image",page)
            attempt = run/"attempt_fixture"
            write_json(attempt/"attempt.json",{"reason":"synthetic fixture, not execution"})
            write_json(attempt/"training.status.json",{"status":"SUCCESS"})
            finalize(run)
            finalize(run)
            record = json.loads((run/"manifest.json").read_text())
            self.assertEqual(len(record["stage_attempts"]["training"]),1)

    def test_report_uses_serialized_impact_fields_and_keeps_repetitions(self):
        from cert_recovery.pilot import finalize, preserved_hashes
        from cert_recovery.evaluation import paired_cost_records
        config = deepcopy(self.config)
        config["execution"]["allow_research_experiments"] = True
        rows = [{**run_workflow_case(workflow_cases()[0],policy,config,manual=True),
                 "repetition":repeat,"outer_elapsed_seconds":.01}
                for repeat in range(2) for policy in ("A_full_recomputation","proposed")]
        self.assertEqual(len(paired_cost_records(rows)),2)
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            write_json(run/"config.json",{"project_root":str(ROOT)})
            write_json(run/"manifest.json",{"stages":{},"preserved_hashes":preserved_hashes(ROOT),"limitations":[]})
            write_json(run/"metrics/workflow_benchmark.json",{"records":rows})
            summary = finalize(run)
            self.assertEqual(summary["system"]["proposed"]["invalidated_certificates"],2)
            self.assertEqual(summary["system"]["proposed"]["recovery_restored_artifacts"],8)

    def test_existing_model_results_are_not_overwritten_or_reinferred(self):
        from cert_recovery.model_pipeline import run_model_evaluation, run_pretrained_baseline
        with tempfile.TemporaryDirectory() as temp:
            config = deepcopy(self.config)
            config["paths"]["results"] = temp
            config["execution"]["allow_model_inference"] = True
            for name in ("finetuned_evaluation.json","pretrained_selection.json"):
                write_json(Path(temp)/name,{"preserve":True})
            with patch.dict("sys.modules",{"torch":None,"transformers":None}):
                with self.assertRaises(FileExistsError):
                    run_model_evaluation(config,manual=True,selection_frozen=True)
                with self.assertRaises(FileExistsError):
                    run_pretrained_baseline(config,manual=True,include_test=True,selection_frozen=True)
            self.assertEqual(json.loads((Path(temp)/"finetuned_evaluation.json").read_text()),{"preserve":True})

    def test_exported_colab_source_records_unknown_git_without_blocking_preparation(self):
        from cert_recovery.pilot import create_run
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/"config").mkdir()
            for name in ("baseline_tests.json","config/config.yaml","config/pinned_smoke_models.json"):
                (root/name).write_bytes((ROOT/name).read_bytes())
            with patch("cert_recovery.pilot.subprocess.run",return_value=SimpleNamespace(returncode=128,stdout="")):
                run = create_run(root)
            manifest = json.loads((run/"manifest.json").read_text())
            self.assertIsNone(manifest["git_revision"])
            self.assertIn("unknown",manifest["git_provenance"])
            self.assertTrue(all(s["status"] == "NOT RUN" for s in manifest["stages"].values()))


if __name__ == "__main__":
    unittest.main()
