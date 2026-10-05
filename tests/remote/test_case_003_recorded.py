"""Replay the historical case-003 HTML; never submit new inference requests."""

import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from remote_model_client import RemoteModelClient, normalize_output

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).parent / "fixtures/case_003_recorded_response.json"


class RecordedCase003Tests(unittest.TestCase):
    def setUp(self):
        for method in ("connect", "connect_ex"):
            guard = patch("socket.socket." + method, side_effect=AssertionError("Offline replay prohibits network"))
            guard.start()
            self.addCleanup(guard.stop)
        self.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.record = self.fixture["record"]

    def test_recorded_response_parses_faithfully_without_changing_annotation(self):
        self.assertEqual(self.fixture["fixture_type"], "recorded_response_not_new_inference")
        self.assertEqual(hashlib.sha256((ROOT / "baseline_tests.json").read_bytes()).hexdigest(),
                         self.fixture["dataset_sha256"])
        dataset = json.loads((ROOT / "baseline_tests.json").read_text(encoding="utf-8"))
        case = next(case for case in dataset["cases"] if case["id"] == "nli-003")
        self.assertEqual(case["input"], self.record["input"])
        self.assertEqual(case["expected_label"], "neutral")
        parsed = normalize_output(self.record["profile"], self.record["output"])
        self.assertEqual(parsed, self.record["normalized_output"])
        self.assertEqual(parsed["original_label"], "Contradiction")
        self.assertEqual(parsed["label"], "contradiction")
        self.assertEqual(parsed["confidence_text"], "99.57%")
        self.assertAlmostEqual(parsed["confidence"], .9957)
        self.assertEqual(parsed["confidence_origin"], "rounded_model_softmax")
        self.assertEqual(parsed["branch"], "model")
        self.assertEqual(parsed["branch_evidence"], "inferred_from_service_label_and_reviewed_source")
        self.assertFalse(parsed["branch_attested"])
        self.assertEqual(parsed["nltk_features"], {"token_overlap": .3333, "wordnet_similarity": .1947})
        self.assertIsNone(parsed["probabilities"])

    def test_successful_transport_and_classification_preserve_high_confidence_mismatch(self):
        remote = RemoteModelClient(self.record["profile"])
        remote.client = Mock()
        remote.client.submit.return_value.result.return_value = self.record["output"]
        case = {"id": "nli-003", "input": self.record["input"], "expected_label": "neutral"}
        replay = remote.infer(case, run_id="offline-recorded-response-replay")
        remote.client.submit.assert_called_once_with(
            api_name="/contradiction_detector", sentence_a="A man is eating pizza.",
            sentence_b="The man owns a bicycle.")
        for field in ("execution_status", "transport_status", "parsing_status", "classification_status"):
            self.assertEqual(replay[field], "SUCCESS")
        self.assertFalse(replay["matches_expected_label"])
        self.assertEqual(replay["output"], self.record["output"])
        self.assertEqual(replay["normalized_output"], self.record["normalized_output"])
        self.assertIsNone(replay["error"])

    def test_serialization_preserves_recorded_provenance_and_unknown_identity(self):
        saved = json.loads(json.dumps(self.fixture, allow_nan=False))
        self.assertEqual(saved, self.fixture)
        self.assertEqual(saved["source_log_sha256"],
                         "da832235a1adae89a6d6d46b6d82c028e21131136c1583d7e0fe50f1208a1798")
        for field in ("verified_loaded_model_id", "loaded_model_revision"):
            self.assertIsNone(saved["record"][field])
        self.assertEqual(saved["record"]["model_identity_status"], "UNKNOWN")
        self.assertFalse(saved["record"]["matches_expected_label"])
        self.assertEqual(saved["record"]["normalized_output"]["confidence_text"], "99.57%")
