import importlib
import unittest

from cert_recovery.config import load_config
from cert_recovery.data_pipeline import clean_rows, grouped_split, normalize_text, prepare_data
from cert_recovery.evaluation import (build_arithmetic_workflow, run_prepared_benchmark, run_integrity_experiment,
                                     run_workflow_case, run_direct_invalidation_experiment,
                                     run_replacement_experiment, run_fault_experiment)
from cert_recovery.model_pipeline import run_finetuning, run_model_evaluation, run_pretrained_baseline, select_threshold
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows():
    return [{"pair_id": f"p{i}", "world_id": f"w{i // 2}", "premise": f"Source {i}",
             "hypothesis": f"Claim {i}", "label": i % 2, "source": "unit-fixture"} for i in range(12)]


class PreparationTests(unittest.TestCase):
    def test_grouped_split_is_reproducible_and_disjoint(self):
        cleaned, _ = clean_rows(rows())
        first = grouped_split(cleaned, (.7, .15, .15), 42)
        self.assertEqual(first, grouped_split(cleaned, (.7, .15, .15), 42))
        groups = [{row["world_id"] for row in first[name]} for name in ("train", "validation", "test")]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        self.assertEqual(sum(map(len, first.values())), len(cleaned))

    def test_cleaning_rejects_conflicts_and_cross_world_repeated_pairs(self):
        first = rows()[0]
        duplicate = {**first, "pair_id": "duplicate"}
        cleaned, counts = clean_rows([first, duplicate])
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(counts["removed_duplicates"], 1)
        with self.assertRaises(ValueError):
            clean_rows([first, {**duplicate, "label": 1}])
        with self.assertRaises(ValueError):
            clean_rows([first, {**duplicate, "world_id": "other"}])
        self.assertEqual(normalize_text("  cafe\u0301  text\n"), "café text")

    def test_every_workload_is_guarded_without_importing_model_packages(self):
        config = load_config(ROOT / "config/config.yaml")
        for function in (prepare_data, run_finetuning, run_model_evaluation, run_pretrained_baseline,
                         run_prepared_benchmark, run_integrity_experiment, run_direct_invalidation_experiment,
                         run_replacement_experiment, run_fault_experiment):
            with self.subTest(function=function.__name__), self.assertRaises(RuntimeError):
                function(config)
            with self.subTest(function=function.__name__), self.assertRaises(RuntimeError):
                function(config, manual=True)
        with self.assertRaises(RuntimeError):
            run_workflow_case({}, "proposed", config, manual=True)

    def test_arithmetic_adapter_core_contract_without_running_experiment(self):
        config = load_config(ROOT / "config/config.yaml")
        case = {"topology": "cascade", "sources": [{"value": 2}, {"value": 3}]}
        system, decisions = build_arithmetic_workflow(case, config["recovery"])
        self.assertEqual(decisions, ["A0", "A1"])
        self.assertEqual(system.latest("Q1").payload, {"value": 5})
        report = system.change_dependency("D0", {"value": 4})
        outcome = system.execute_recovery(system.plan_recovery(report))
        self.assertEqual(outcome.status, "COMPLETE")
        self.assertEqual(system.latest("Q1").payload, {"value": 7})
        self.assertTrue(system.is_valid(system.latest("A1").ref))

    def test_test_evaluation_requires_frozen_selection_before_import(self):
        config = load_config(ROOT / "config/config.yaml")
        config["execution"]["allow_model_inference"] = True
        with self.assertRaisesRegex(RuntimeError, "Freeze"):
            run_model_evaluation(config, manual=True)
        with self.assertRaisesRegex(RuntimeError, "Freeze"):
            run_pretrained_baseline(config, manual=True, include_test=True)

    def test_threshold_uses_only_supplied_validation_fixture(self):
        self.assertEqual(select_threshold([0, 1], [.1, .8], [.2, .5, .9]), .2)
        with self.assertRaises(ValueError):
            select_threshold([], [], [.5])

    def test_importing_prepared_modules_has_no_workload_side_effects(self):
        for module in ("cert_recovery.model_pipeline", "cert_recovery.data_pipeline", "cert_recovery.evaluation"):
            self.assertIsNotNone(importlib.import_module(module))
