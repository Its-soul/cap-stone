"""Offline workflow tests. Fake transports here are never written as live results."""

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from remote_model_client import (
    PROFILES, RESULT_SCHEMA_VERSION, RemoteModelClient, RemoteModelError,
    ResponseValidationError, normalize_output, redact,
)
from run_baseline import ROOT, create_run_file, load_dataset, run, save_current_run


def html_result(label="Consistent", confidence="98.75%", overlap=None, similarity=None):
    html = f"<div>{label}</div><div>Confidence: <strong>{confidence}</strong></div>"
    if overlap is not None:
        html += f"<div><b>Token Overlap:</b> {overlap}<br></div>"
    if similarity is not None:
        html += f"<div><b>WordNet Similarity:</b> {similarity}<br></div>"
    return html


def cpu_result(label="entailment"):
    return ({"label": label, "confidences": [
        {"label": name, "confidence": .9 if name == label.lower() else .05}
        for name in ("entailment", "contradiction", "neutral")]}, "")


class RemoteWorkflowTests(unittest.TestCase):
    def setUp(self):
        # Fail any accidental live connection, including calls hidden in a client.
        guard = patch("socket.socket.connect", side_effect=AssertionError("Offline tests prohibit network access"))
        guard.start()
        self.addCleanup(guard.stop)

    def infer_response(self, response=None, error=None, profile="nli-zero-gpu"):
        remote = RemoteModelClient(profile)
        remote.client = Mock()
        job = remote.client.submit.return_value
        job.result.return_value = response
        job.result.side_effect = error
        case = load_dataset(ROOT / "baseline_tests.json")[0]["cases"][0]
        return remote.infer(case, run_id="offline-fixture"), remote

    def test_dataset_is_fixed_and_expected_labels_are_not_api_arguments(self):
        dataset, digest = load_dataset(ROOT / "baseline_tests.json")
        self.assertEqual(len(dataset["cases"]), 6)
        self.assertEqual(digest, load_dataset(ROOT / "baseline_tests.json")[1])
        remote = RemoteModelClient()
        remote.client = Mock()
        remote.client.submit.return_value.result.return_value = html_result()
        record = remote.infer(dataset["cases"][0])
        remote.client.submit.assert_called_once_with(
            api_name="/contradiction_detector", sentence_a="A man is eating pizza.",
            sentence_b="A man is eating food.")
        self.assertEqual(record["execution_status"], "SUCCESS")
        self.assertTrue(record["matches_expected_label"])

    def test_error_text_is_not_classified_as_success(self):
        remote = RemoteModelClient()
        remote.client = Mock()
        remote.client.submit.return_value.result.return_value = "Quota exceeded. Please try later."
        case = load_dataset(ROOT / "baseline_tests.json")[0]["cases"][0]
        record = remote.infer(case)
        self.assertEqual(record["execution_status"], "ERROR")
        self.assertEqual(record["output"], "Quota exceeded. Please try later.")
        self.assertIsNone(record["normalized_output"])

    def test_html_mapping_does_not_invent_full_probabilities(self):
        result = normalize_output("nli-zero-gpu", html_result("Contradiction"))
        self.assertEqual(result["label"], "contradiction")
        self.assertEqual(result["confidence"], .9875)
        self.assertIsNone(result["probabilities"])
        record, _ = self.infer_response(html_result("Contradiction"))
        self.assertEqual(record["classification_status"], "SUCCESS")
        self.assertFalse(record["matches_expected_label"])

    def test_cpu_response_requires_valid_three_class_distribution(self):
        raw = ({"label": "entailment", "confidences": [
            {"label": "entailment", "confidence": .9},
            {"label": "neutral", "confidence": .08},
            {"label": "contradiction", "confidence": .02}]}, "")
        self.assertEqual(normalize_output("nli-cpu", raw)["label"], "entailment")
        raw[0]["confidences"][0]["confidence"] = 1.9
        with self.assertRaises(RemoteModelError):
            normalize_output("nli-cpu", raw)

    def test_alternative_model_requires_explicit_opt_in_before_network(self):
        remote = RemoteModelClient()
        remote.api = Mock()
        with self.assertRaisesRegex(RemoteModelError, "alternative-model"):
            remote.connect()
        remote.api.space_info.assert_not_called()

    def test_unavailable_space_is_rejected_before_inference(self):
        remote = RemoteModelClient()
        remote.api = Mock()
        runtime = SimpleNamespace(stage="PAUSED", hardware=None, raw={})
        remote.api.space_info.return_value = SimpleNamespace(
            runtime=runtime, sha=remote.profile.reviewed_space_revision, host="https://example.hf.space")
        with self.assertRaisesRegex(RemoteModelError, "PAUSED"):
            remote.connect(allow_alternative_model=True)
        self.assertIsNone(remote.client)

    def test_revision_drift_is_rejected(self):
        remote = RemoteModelClient()
        remote.api = Mock()
        runtime = SimpleNamespace(stage="RUNNING", hardware="zero-a10g", raw={})
        remote.api.space_info.return_value = SimpleNamespace(runtime=runtime, sha="changed", host="https://example.hf.space")
        with self.assertRaisesRegex(RemoteModelError, "revision"):
            remote.connect(allow_alternative_model=True)

    def test_schema_drift_is_rejected_before_submission(self):
        remote = RemoteModelClient()
        remote.api = Mock()
        runtime = SimpleNamespace(stage="RUNNING", hardware="zero-a10g", raw={})
        remote.api.space_info.return_value = SimpleNamespace(
            runtime=runtime, sha=remote.profile.reviewed_space_revision, host="https://example.hf.space", sdk="gradio")
        with patch("remote_model_client.Client") as client:
            client.return_value.view_api.return_value = {"named_endpoints": {
                remote.profile.api_name: {"parameters": [], "returns": []}}}
            with self.assertRaisesRegex(RemoteModelError, "parameters"):
                remote.connect(allow_alternative_model=True)
            client.return_value.submit.assert_not_called()

    def test_timeout_is_recorded_without_retry_even_if_cancellation_fails(self):
        remote = RemoteModelClient(timeout=1)
        remote.client = Mock()
        job = remote.client.submit.return_value
        job.result.side_effect = TimeoutError()
        job.cancel.side_effect = RuntimeError("Cancellation failed")
        case = load_dataset(ROOT / "baseline_tests.json")[0]["cases"][0]
        record = remote.infer(case)
        self.assertEqual(record["execution_status"], "TIMEOUT")
        self.assertIn("remote outcome unknown", record["error"]["message"])
        remote.client.submit.assert_called_once()

    def test_results_are_exclusive_and_previous_run_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            first = create_run_file(Path(directory))
            save_current_run(first, {"status": "SUCCESS", "output": "first"})
            before = first.read_bytes()
            second = create_run_file(Path(directory))
            save_current_run(second, {"status": "FAILED", "error": "second"})
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_bytes(), before)
            self.assertEqual(len(list(Path(directory).glob("baseline_*.json"))), 2)

    def test_failed_preflight_saves_every_case_as_not_run(self):
        with tempfile.TemporaryDirectory() as directory, patch("run_baseline.RemoteModelClient") as factory:
            remote = factory.return_value
            remote.metadata = {"profile": "nli-zero-gpu"}
            remote.profile = PROFILES["nli-zero-gpu"]
            remote.connect.side_effect = RuntimeError("Space unavailable")
            args = argparse.Namespace(results_dir=Path(directory), dataset=ROOT / "baseline_tests.json",
                profile="nli-zero-gpu", allow_alternative_model=True, timeout=1, limit=None)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(run(args), 1)
            saved = json.loads(next(Path(directory).glob("baseline_*.json")).read_text())
            self.assertEqual(saved["status"], "FAILED")
            self.assertEqual(saved["summary"], {"NOT_RUN": 6})
            self.assertTrue(all(row["output"] is None for row in saved["results"]))
            remote.infer.assert_not_called()

    def test_token_is_redacted_recursively(self):
        token = "hf_" + "a" * 32
        self.assertEqual(redact({"errors": ["token=" + token]}, token), {"errors": ["token=[REDACTED]"]})

    def test_exhausted_included_quota_stops_before_inference(self):
        with tempfile.TemporaryDirectory() as directory, patch("run_baseline.RemoteModelClient") as factory:
            remote = factory.return_value
            remote.metadata = {"profile": "nli-zero-gpu"}
            remote.profile = PROFILES["nli-zero-gpu"]
            remote.connect.return_value = remote.metadata
            remote.quota.return_value = {"status": "AVAILABLE", "remaining_gpu_seconds": 0}
            args = argparse.Namespace(results_dir=Path(directory), dataset=ROOT / "baseline_tests.json",
                profile="nli-zero-gpu", allow_alternative_model=True, timeout=1, limit=1)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(run(args), 1)
            saved = json.loads(next(Path(directory).glob("baseline_*.json")).read_text())
            self.assertEqual(saved["status"], "FAILED")
            self.assertIn("quota is exhausted", saved["error"]["message"])
            remote.infer.assert_not_called()

    def test_consistent_is_entailment_only_under_the_wrapped_contract(self):
        result = normalize_output("nli-zero-gpu", html_result("✅ Consistent"))
        self.assertEqual(result["original_label"], "Consistent")
        self.assertEqual(result["label"], "entailment")
        self.assertEqual(result["branch"], "model")
        self.assertIn("inferred", result["branch_evidence"])
        self.assertFalse(result["branch_attested"])
        self.assertTrue(result["advisory_only"])
        with self.assertRaisesRegex(ResponseValidationError, "Unknown service label"):
            normalize_output("nli-cpu", cpu_result("Consistent"))

    def test_unknown_profile_has_no_implicit_cpu_adapter(self):
        with self.assertRaises(ResponseValidationError) as raised:
            normalize_output("unreviewed-space", cpu_result())
        self.assertEqual(raised.exception.code, "unknown_profile")
        with self.assertRaises(RemoteModelError):
            RemoteModelClient("unreviewed-space")

    def test_model_neutral_is_inferred_from_returned_features(self):
        for overlap, similarity in ((.1, .05), (0, .12), (0, .3)):
            with self.subTest(overlap=overlap, similarity=similarity):
                result = normalize_output("nli-zero-gpu", html_result("Unrelated", overlap=overlap, similarity=similarity))
                self.assertEqual(result["label"], "neutral")
                self.assertEqual(result["original_label"], "Unrelated")
                self.assertEqual(result["branch"], "model")
                self.assertIn("inferred_from_returned_nltk_features", result["branch_evidence"])
                self.assertEqual(result["confidence_origin"], "rounded_model_softmax")

    def test_heuristic_unrelated_is_not_model_neutral(self):
        raw = html_result("Unrelated", confidence="95.00%", overlap=0, similarity=.05)
        record, _ = self.infer_response(raw)
        result = record["normalized_output"]
        self.assertEqual(result["branch"], "heuristic")
        self.assertIsNone(result["label"])
        self.assertEqual(result["confidence_origin"], "heuristic_score")
        self.assertEqual(result["confidence_text"], "95.00%")
        self.assertEqual(result["confidence"], .95)
        self.assertEqual(record["classification_status"], "HEURISTIC")
        self.assertIsNone(record["matches_expected_label"])
        self.assertEqual(record["output"], raw)

    def test_ambiguous_unrelated_keeps_branch_and_confidence_origin_unknown(self):
        for kwargs in ({}, {"overlap": 0}, {"similarity": .05}):
            with self.subTest(features=kwargs):
                record, _ = self.infer_response(html_result("Unrelated", **kwargs))
                result = record["normalized_output"]
                self.assertIsNone(result["label"])
                self.assertEqual(result["branch"], "unknown")
                self.assertEqual(result["branch_evidence"], "unknown")
                self.assertEqual(result["confidence_origin"], "unknown")
                self.assertEqual(record["classification_status"], "UNKNOWN")
                self.assertIsNone(record["matches_expected_label"])

    def test_html_confidence_is_rounded_advice_with_no_distribution(self):
        result = normalize_output("nli-zero-gpu", html_result())
        self.assertEqual(result["confidence_text"], "98.75%")
        self.assertEqual(result["confidence"], .9875)
        self.assertEqual(result["confidence_origin"], "rounded_model_softmax")
        self.assertIsNone(result["probabilities"])
        self.assertNotIn("calibrated", result)

    def test_cpu_neutral_preserves_only_returned_probabilities(self):
        raw = cpu_result("NEUTRAL")
        result = normalize_output("nli-cpu", raw)
        self.assertEqual(result["original_label"], "NEUTRAL")
        self.assertEqual(result["label"], "neutral")
        self.assertEqual(result["confidence_origin"], "model_softmax")
        self.assertIsNone(result["confidence_text"])
        self.assertEqual(result["probabilities"], {row["label"]: row["confidence"] for row in raw[0]["confidences"]})

    def test_unknown_and_missing_html_labels_are_explicit_errors(self):
        for raw, code in ((html_result("Independent"), "unknown_label"),
                          ("<div>Confidence: <strong>98.75%</strong></div>", "missing_label"),
                          ("<div></div>", "missing_label")):
            with self.subTest(raw=raw):
                record, _ = self.infer_response(raw)
                self.assertEqual(record["transport_status"], "SUCCESS")
                self.assertEqual(record["parsing_status"], "ERROR")
                self.assertEqual(record["classification_status"], "ERROR")
                self.assertEqual(record["error"]["code"], code)
                self.assertEqual(record["output"], raw)

    def test_missing_confidence_is_rejected_and_label_observation_is_retained(self):
        record, _ = self.infer_response("<div>Consistent</div>")
        self.assertEqual(record["error"]["code"], "missing_confidence")
        self.assertEqual(record["error"]["observations"]["original_label"], "Consistent")

    def test_malformed_html_is_rejected_without_repair(self):
        for raw in ("<div>Consistent<div>Confidence: 98.75%</div>",
                    "<div><strong>Consistent</div></strong>",
                    '<div class="broken>Consistent</div>',
                    "<script>Consistent</script><div>Confidence: 98.75%</div>",
                    html_result() + "<div>Contradiction</div>"):
            with self.subTest(raw=raw):
                record, _ = self.infer_response(raw)
                self.assertEqual(record["execution_status"], "ERROR")
                self.assertEqual(record["transport_status"], "SUCCESS")
                self.assertIsNone(record["normalized_output"])
                self.assertEqual(record["output"], raw)

    def test_invalid_displayed_confidence_values_are_rejected(self):
        for confidence in ("NaN%", "inf%", "-inf%", "-1%", "100.01%", "abc%", "0.9", ""):
            with self.subTest(confidence=confidence):
                record, _ = self.infer_response(html_result(confidence=confidence))
                self.assertEqual(record["parsing_status"], "ERROR")
                self.assertEqual(record["error"]["code"], "invalid_confidence" if confidence else "missing_confidence")
                json.dumps(record, allow_nan=False)

    def test_displayed_confidence_permitted_range_includes_endpoints(self):
        for text, value in (("0.00%", 0), ("100.00%", 1)):
            with self.subTest(text=text):
                self.assertEqual(normalize_output("nli-zero-gpu", html_result(confidence=text))["confidence"], value)

    def test_bad_nltk_features_and_inconsistent_filter_label_are_rejected(self):
        for kwargs in ({"overlap": "NaN", "similarity": .05},
                       {"overlap": 0, "similarity": 1.1},
                       {"overlap": 0, "similarity": -.1},
                       {"overlap": 0, "similarity": .05}):
            with self.subTest(features=kwargs):
                record, _ = self.infer_response(html_result("Consistent", **kwargs))
                self.assertEqual(record["parsing_status"], "ERROR")

    def test_empty_input_warning_is_not_a_classification(self):
        raw = "⚠️ Please enter both sentences."
        record, _ = self.infer_response(raw)
        self.assertEqual(record["error"]["code"], "empty_input_warning")
        self.assertEqual(record["transport_status"], "SUCCESS")
        self.assertEqual(record["parsing_status"], "ERROR")
        self.assertEqual(record["output"], raw)
        self.assertIsNone(record["matches_expected_label"])

    def test_quota_exception_is_transport_failure_without_retry(self):
        record, remote = self.infer_response(error=RuntimeError("You have exceeded your GPU quota"))
        self.assertEqual(record["error"]["code"], "quota_exceeded")
        self.assertEqual(record["error"]["stage"], "transport")
        self.assertEqual(record["transport_status"], "ERROR")
        self.assertEqual(record["parsing_status"], "NOT_RUN")
        self.assertIsNone(record["output"])
        remote.client.submit.assert_called_once()

    def test_unavailable_exception_is_transport_failure(self):
        record, _ = self.infer_response(error=RuntimeError("503 Space unavailable"))
        self.assertEqual(record["error"]["code"], "space_unavailable")
        self.assertEqual(record["transport_status"], "ERROR")
        self.assertEqual(record["classification_status"], "NOT_RUN")

    def test_timeout_keeps_parsing_not_run_and_remote_outcome_unknown(self):
        record, remote = self.infer_response(error=TimeoutError("delayed"))
        self.assertEqual(record["transport_status"], "TIMEOUT")
        self.assertEqual(record["parsing_status"], "NOT_RUN")
        self.assertEqual(record["error"]["code"], "timeout")
        remote.client.submit.return_value.cancel.assert_called_once()
        self.assertTrue(record["cancellation_requested"])
        remote.client.submit.side_effect = TimeoutError("submission stalled")
        record = remote.infer(load_dataset(ROOT / "baseline_tests.json")[0]["cases"][0])
        self.assertFalse(record["cancellation_requested"])
        self.assertIn("no job handle", record["error"]["message"])

    def test_provenance_never_attests_loaded_model_from_configuration(self):
        record, remote = self.infer_response(html_result())
        self.assertEqual(record["schema_version"], RESULT_SCHEMA_VERSION)
        self.assertEqual(record["run_id"], "offline-fixture")
        self.assertEqual(record["test_id"], "nli-001")
        self.assertTrue(record["timestamp"].endswith("+00:00"))
        self.assertEqual(record["profile"], "nli-zero-gpu")
        self.assertEqual(record["reviewed_space_revision"], PROFILES["nli-zero-gpu"].reviewed_space_revision)
        self.assertEqual(record["configured_model_id"], "cross-encoder/nli-deberta-v3-small")
        self.assertEqual(record["fallback_model_id"], PROFILES["nli-zero-gpu"].model_id)
        for value in (record, remote.metadata):
            self.assertIsNone(value["verified_loaded_model_id"])
            self.assertIsNone(value["loaded_model_revision"])
            self.assertEqual(value["model_identity_status"], "UNKNOWN")
        self.assertIn("fine-tuned-model", remote.metadata["source_model_candidates"])
        self.assertGreaterEqual(record["latency_seconds"], 0)
        self.assertIn("service", record["latency_scope"])

    def test_observed_hub_revision_is_not_loaded_weight_attestation(self):
        remote = RemoteModelClient()
        remote.api = Mock()
        remote.api.space_info.return_value = SimpleNamespace(
            runtime=SimpleNamespace(stage="RUNNING", hardware="zero-a10g", raw={}),
            sha=remote.profile.reviewed_space_revision, host="https://example.hf.space", sdk="gradio")
        remote.api.model_info.return_value = SimpleNamespace(sha="hub-revision-only")
        with patch("remote_model_client.Client") as client:
            client.return_value.view_api.return_value = {"named_endpoints": {remote.profile.api_name: {
                "parameters": [{"parameter_name": name, "type": {"type": "string"}}
                               for name in remote.profile.parameter_names],
                "returns": [{"component": "Html"}]}}}
            client.return_value.config = {"dependencies": [{"api_name": "contradiction_detector", "zerogpu": True}]}
            metadata = remote.connect(allow_alternative_model=True)
            self.assertEqual(metadata["model_hub_revision_observed"], "hub-revision-only")
            self.assertIsNone(metadata["verified_loaded_model_id"])
            self.assertIsNone(metadata["loaded_model_revision"])
            client.return_value.submit.assert_not_called()

    def test_bad_cpu_responses_fail_explicitly(self):
        fixtures = [(None, ""), ({}, ""), ({"label": "neutral"}, ""),
                    ({"label": "neutral", "confidences": [None, {}, {}]}, "")]
        duplicate = cpu_result()
        duplicate[0]["confidences"][1]["label"] = "entailment"
        fixtures.append(duplicate)
        inconsistent = cpu_result()
        inconsistent[0]["label"] = "neutral"
        fixtures.append(inconsistent)
        bad_sum = cpu_result()
        bad_sum[0]["confidences"][0]["confidence"] = .8
        fixtures.append(bad_sum)
        for raw in fixtures:
            with self.subTest(raw=raw):
                record, _ = self.infer_response(raw, profile="nli-cpu")
                self.assertEqual(record["transport_status"], "SUCCESS")
                self.assertEqual(record["parsing_status"], "ERROR")
                self.assertEqual(record["error"]["type"], "ResponseValidationError")

    def test_invalid_cpu_confidence_is_logged_as_strict_json(self):
        for value in (float("nan"), float("inf"), -float("inf"), -.1, 1.1, True, "0.9"):
            with self.subTest(value=value):
                raw = cpu_result()
                raw[0]["confidences"][0]["confidence"] = value
                record, _ = self.infer_response(raw, profile="nli-cpu")
                self.assertEqual(record["error"]["code"], "invalid_confidence")
                encoded = json.dumps(record, allow_nan=False)
                if isinstance(value, float) and value != value:
                    self.assertIn('"nonfinite_number": "nan"', encoded)

    def test_raw_cpu_response_is_preserved_in_json_compatible_form(self):
        raw = cpu_result()
        record, _ = self.infer_response(raw, profile="nli-cpu")
        self.assertEqual(record["output"], list(raw))
        self.assertEqual(json.loads(json.dumps(record))["output"], list(raw))
        self.assertEqual(record["transport_status"], "SUCCESS")
        self.assertEqual(record["parsing_status"], "SUCCESS")
        self.assertEqual(record["classification_status"], "SUCCESS")

    def test_new_schema_run_serializes_and_does_not_rewrite_legacy_log(self):
        record, remote = self.infer_response(html_result("Unrelated"))
        remote.close = Mock()
        remote.quota = Mock(return_value={"status": "UNKNOWN", "remaining_gpu_seconds": None})
        remote.connect = Mock(return_value=remote.metadata)
        with tempfile.TemporaryDirectory() as directory, patch("run_baseline.RemoteModelClient", return_value=remote):
            legacy = create_run_file(Path(directory))
            legacy_document = {"schema_version": 1, "run_id": legacy.stem, "status": "SUCCESS",
                               "results": [{"output": "legacy", "normalized_output": {"label": "neutral"}}]}
            save_current_run(legacy, legacy_document)
            before = legacy.read_bytes()
            args = argparse.Namespace(results_dir=Path(directory), dataset=ROOT / "baseline_tests.json",
                profile="nli-zero-gpu", allow_alternative_model=True, timeout=1, limit=1)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(run(args), 0)
            new_path = next(path for path in Path(directory).glob("baseline_*.json") if path != legacy)
            saved = json.loads(new_path.read_text())
            self.assertEqual(saved["schema_version"], 2)
            self.assertEqual(saved["results"][0]["run_id"], saved["run_id"])
            self.assertEqual(saved["classification_summary"], {"UNKNOWN": 1})
            self.assertEqual(saved["results"][0]["output"], html_result("Unrelated"))
            self.assertEqual(legacy.read_bytes(), before)

    def test_not_run_cases_have_full_provenance_and_no_claimed_transport(self):
        with tempfile.TemporaryDirectory() as directory, patch("run_baseline.RemoteModelClient") as factory:
            remote = factory.return_value
            remote.metadata = {"profile": "nli-zero-gpu"}
            remote.connect.side_effect = RuntimeError("Space unavailable")
            args = argparse.Namespace(results_dir=Path(directory), dataset=ROOT / "baseline_tests.json",
                profile="nli-zero-gpu", allow_alternative_model=True, timeout=1, limit=1)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(run(args), 1)
            saved = json.loads(next(Path(directory).glob("baseline_*.json")).read_text())
            record = saved["results"][0]
            self.assertEqual(record["run_id"], saved["run_id"])
            for name in ("transport_status", "parsing_status", "classification_status"):
                self.assertEqual(record[name], "NOT_RUN")
            self.assertIsNone(record["verified_loaded_model_id"])
            self.assertIsNone(record["loaded_model_revision"])
            self.assertIsNone(record["latency_seconds"])
            remote.infer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
