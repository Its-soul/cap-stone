"""Synthetic schema-2 fixtures only: no live requests or benchmark output files."""

from dataclasses import replace
import json
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import Mock, patch

# Reuse the core's exact-value/rebuilding fixture, not a parallel recovery engine.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tests" / "remote"))
from test_engine import copy_value, exact_value, make_system
from test_remote_workflow import html_result

from cert_recovery import ArtifactKind as K, CertificateStatus as S, CertificateSystem, Ref
from cert_recovery.provenance import AuditJournal
from remote_model_client import RemoteModelClient
from semantic_advice import SemanticAdviceAdapter


def fixture_response(prepared, raw=None, error=None):
    """Exercise the existing schema-2 handler through a mocked transport."""
    remote = RemoteModelClient(prepared.profile)
    remote.client = Mock()
    remote.client.submit.return_value.result.return_value = html_result() if raw is None else raw
    remote.client.submit.return_value.result.side_effect = error
    record = remote.infer(prepared.to_case(), run_id=prepared.request_id)
    record["fixture"] = "synthetic mocked response; not remote execution evidence"
    return record


class SemanticAdviceIntegrationTests(unittest.TestCase):
    def setUp(self):
        for method in ("connect", "connect_ex"):
            guard = patch(f"socket.socket.{method}", side_effect=AssertionError("Offline integration tests prohibit network access"))
            guard.start()
            self.addCleanup(guard.stop)
        self.system, self.d, self.e, self.q, self.c, self.a = make_system()
        self.adapter = SemanticAdviceAdapter(self.system, lambda node: f"Value is {node.payload['value']}.")

    def assert_audit(self, annotation):
        event = self.system.audit_export()[-1]
        self.assertEqual(event["kind"], "semantic_advice")
        self.assertEqual(event["payload"], annotation)
        self.assertTrue(AuditJournal.verify(self.system.audit_export(), self.system.checkpoint()))
        json.dumps(annotation, allow_nan=False)

    def test_fresh_model_advice_is_bound_annotation_with_valid_support(self):
        prepared = self.adapter.prepare(self.d, self.q)
        response = fixture_response(prepared)
        nodes, epoch, history = self.system.current_artifacts(), self.system.epoch, self.system.certificate_history()
        annotation = self.adapter.consume(prepared, response)
        self.assertEqual(annotation["disposition"], "ACCEPTED")
        self.assertEqual(annotation["nli_relation"], "entailment")
        self.assertIsNone(annotation["dependency_judgment"])
        self.assertEqual(annotation["input_bindings"], [
            {**ref.to_dict(), "content_hash": self.system.artifact(ref).content_hash} for ref in (self.d, self.q)])
        self.assertEqual(annotation["response"], response)
        self.assertIsNone(annotation["response"]["verified_loaded_model_id"])
        self.assertIsNone(annotation["response"]["loaded_model_revision"])
        self.assertEqual(annotation["normalized_output"]["confidence_origin"], "rounded_model_softmax")
        self.assertFalse(annotation["normalized_output"]["branch_attested"])
        self.assertIsNone(response["matches_expected_label"])
        self.assertEqual(self.system.current_artifacts(), nodes)
        self.assertEqual(self.system.epoch, epoch)
        self.assertEqual(self.system.certificate_history(), history)
        self.assertEqual(self.system.artifact(self.q).supports, (self.e,))
        self.assertEqual(self.system.certificate_status(self.c), S.VALID)
        self.assert_audit(annotation)
        saved_response = json.loads(json.dumps(response))
        response["normalized_output"]["confidence"] = .01
        annotation["response"]["normalized_output"]["confidence"] = .02
        self.assertEqual(self.system.audit_export()[-1]["payload"]["response"], saved_response)
        self.assertTrue(AuditJournal.verify(self.system.audit_export(), self.system.checkpoint()))
        self.assertEqual(self.system.commit_decision(self.a), {"action": "mock-accept"})

    def test_high_confidence_entailment_cannot_pass_failed_verifier(self):
        bad = self.system.add_artifact("Bad", K.CLAIM, {"value": -1}, (self.e,))
        prepared = self.adapter.prepare(self.d, bad)
        annotation = self.adapter.consume(prepared, fixture_response(prepared, html_result(confidence="100.00%")))
        self.assertEqual(annotation["disposition"], "ACCEPTED")
        self.assertEqual(annotation["normalized_output"]["confidence"], 1)
        self.assertIsNone(annotation["dependency_judgment"])
        with self.assertRaisesRegex(ValueError, "Verification failed"):
            self.system.issue_certificate("BadCertificate", bad, "exact-value")
        with self.assertRaises(KeyError):
            self.system.latest("BadCertificate")
        verification = self.system.audit_export()[-1]
        self.assertEqual(verification["kind"], "verification_result")
        self.assertFalse(verification["payload"]["passed"])
        self.assertEqual(verification["payload"]["reason"], "source mismatch")
        self.assertIsNone(verification["payload"]["confidence"])
        self.assertEqual(self.system.artifact(bad).supports, (self.e,))
        self.assertEqual(self.system.certificate_status(self.c), S.VALID)
        self.system.commit_decision(self.a)
        self.assertTrue(AuditJournal.verify(self.system.audit_export(), self.system.checkpoint()))

    def test_change_between_preparation_and_consumption_rejects_exact_old_binding(self):
        prepared = self.adapter.prepare(self.d, self.q)
        response = fixture_response(prepared)
        old_hash = self.system.artifact(self.d).content_hash
        self.system.change_dependency("D", {"value": 9})
        annotation = self.adapter.consume(prepared, response)
        self.assertEqual(annotation["disposition"], "REJECTED_STALE")
        self.assertIsNone(annotation["nli_relation"])
        self.assertEqual(annotation["input_bindings"][0], {"artifact_id": "D", "version": 1, "content_hash": old_hash})
        self.assertEqual(annotation["freshness_failures"], [
            {"artifact_id": "D", "version": 1, "reason": "input_version_changed"},
            {"artifact_id": "Q", "version": 1, "reason": "input_or_ancestry_invalid"}])
        self.assertEqual(self.system.latest("D").ref, Ref("D", 2))
        self.assertEqual(self.system.certificate_status(self.c), S.INVALID)
        self.assertEqual(annotation["response"], response)
        self.assert_audit(annotation)

    def test_unchanged_derived_versions_are_stale_when_ancestry_changes(self):
        prepared = self.adapter.prepare(self.e, self.q)
        response = fixture_response(prepared)
        self.system.change_dependency("D", {"value": 7})
        annotation = self.adapter.consume(prepared, response)
        self.assertEqual(self.system.latest("E").ref, self.e)
        self.assertEqual(self.system.latest("Q").ref, self.q)
        self.assertEqual(annotation["disposition"], "REJECTED_STALE")
        self.assertTrue(all(row["reason"] == "input_or_ancestry_invalid" for row in annotation["freshness_failures"]))
        self.assert_audit(annotation)

    def test_heuristic_and_ambiguous_output_are_ignored_without_changing_edges(self):
        for raw, reason, origin in (
            (html_result("Unrelated", "95.00%", 0, .05), "heuristic_output_is_not_model_advice", "heuristic_score"),
            (html_result("Unrelated"), "ambiguous_advice_branch", "unknown")):
            with self.subTest(reason=reason):
                prepared = self.adapter.prepare(self.d, self.q)
                nodes = self.system.current_artifacts()
                annotation = self.adapter.consume(prepared, fixture_response(prepared, raw))
                self.assertEqual(annotation["disposition"], "IGNORED")
                self.assertEqual(annotation["reason"], reason)
                self.assertIsNone(annotation["nli_relation"])
                self.assertIsNone(annotation["dependency_judgment"])
                self.assertEqual(annotation["normalized_output"]["confidence_origin"], origin)
                self.assertEqual(self.system.current_artifacts(), nodes)
                self.assertEqual(self.system.artifact(self.q).supports, (self.e,))
                self.assert_audit(annotation)
                self.system.commit_decision(self.a)

    def test_timeout_malformed_and_unavailable_advice_do_not_block_deterministic_recovery(self):
        for raw, error, reason in ((None, TimeoutError("synthetic timeout"), "advice_unavailable"),
                                   ("<div>Consistent", None, "malformed_service_response"),
                                   (None, RuntimeError("503 Space unavailable (fixture)"), "advice_unavailable")):
            with self.subTest(reason=reason, error=error):
                system, d, e, q, c, decision = make_system()
                adapter = SemanticAdviceAdapter(system)
                prepared = adapter.prepare(d, q)
                annotation = adapter.consume(prepared, fixture_response(prepared, raw, error))
                self.assertEqual(annotation["disposition"], "IGNORED")
                self.assertEqual(annotation["reason"], reason)
                self.assertIsNone(annotation["dependency_judgment"])
                self.assertEqual(system.certificate_status(c), S.VALID)
                system.commit_decision(decision)
                outcome = system.execute_recovery(system.plan_recovery(system.change_dependency("D", {"value": 8})))
                self.assertEqual(outcome.restored, (Ref("E", 2), Ref("Q", 2), Ref("C", 2), Ref("A", 2)))
                self.assertEqual(system.latest("Q").payload, {"value": 8})
                self.assertEqual(system.certificate_status(c), S.SUPERSEDED)
                self.assertEqual(system.commit_decision(Ref("A", 2))["support"], "C@2")
                self.assertTrue(AuditJournal.verify(system.audit_export(), system.checkpoint()))

    def test_recovery_replaces_affected_certificate_and_preserves_independent_branch(self):
        independent = self.system.add_dependency("U", {"value": 99})
        other = self.system.issue_certificate("Other", independent, "exact-value")
        other_decision = self.system.add_artifact("OtherDecision", K.DECISION, {"action": "independent"}, (other,))
        unrelated_nodes = tuple(self.system.artifact(ref) for ref in (independent, other, other_decision))
        other_history = self.system.certificate_history(other)
        unrelated_prepared = SemanticAdviceAdapter(self.system).prepare(independent, other)
        unrelated_response = fixture_response(unrelated_prepared)
        prepared = self.adapter.prepare(self.d, self.q)
        response = fixture_response(prepared)
        accepted = self.adapter.consume(prepared, response)
        self.assertEqual(accepted["disposition"], "ACCEPTED")
        report = self.system.change_dependency("D", {"value": 9})
        self.assertIn(self.c, report.transitive_certificates)
        self.assertIn(other, report.unaffected_certificates)
        self.assertEqual(self.system.certificate_status(self.c), S.INVALID)
        self.assertEqual(self.system.certificate_status(other), S.VALID)
        with self.assertRaisesRegex(ValueError, "stale/invalid"):
            self.system.commit_decision(self.a)
        self.assertFalse(any(event["kind"] == "decision_committed" for event in self.system.audit_export()))
        self.assertEqual(self.adapter.consume(prepared, response)["disposition"], "REJECTED_STALE")
        outcome = self.system.execute_recovery(self.system.plan_recovery(report))
        self.assertEqual(outcome.restored, (Ref("E", 2), Ref("Q", 2), Ref("C", 2), Ref("A", 2)))
        for name, supports in (("E", (Ref("D", 2),)), ("Q", (Ref("E", 2),)), ("A", (Ref("C", 2),))):
            self.assertEqual(self.system.latest(name).ref, Ref(name, 2))
            self.assertEqual(self.system.latest(name).supports, supports)
        self.assertEqual(self.system.latest("Q").payload, {"value": 9})
        replacement = self.system.certificate(Ref("C", 2))
        self.assertEqual(replacement.parent_certificate, self.c)
        self.assertEqual(replacement.subject, Ref("Q", 2))
        self.assertEqual(set(replacement.snapshot), {Ref("D", 2), Ref("E", 2), Ref("Q", 2)})
        self.assertEqual(self.system.certificate_status(Ref("C", 2)), S.VALID)
        self.assertEqual([event.current for event in self.system.certificate_history(self.c)],
                         [S.VALID, S.INVALID, S.PENDING_REVERIFICATION, S.SUPERSEDED])
        self.assertEqual(tuple(self.system.artifact(node.ref) for node in unrelated_nodes), unrelated_nodes)
        self.assertEqual(self.system.certificate_history(other), other_history)
        self.assertEqual(self.adapter.consume(unrelated_prepared, unrelated_response)["disposition"], "ACCEPTED")
        with self.assertRaisesRegex(ValueError, "stale/invalid"):
            self.system.commit_decision(self.a)
        with self.assertRaisesRegex(ValueError, "stale or invalid"):
            self.system.add_artifact("OldSupportDecision", K.DECISION, {}, (self.c,))
        new_prepared = self.adapter.prepare(Ref("D", 2), Ref("Q", 2))
        new_annotation = self.adapter.consume(new_prepared, fixture_response(new_prepared))
        self.assertEqual(new_annotation["disposition"], "ACCEPTED")
        self.assertEqual([binding["version"] for binding in new_annotation["input_bindings"]], [2, 2])
        self.assert_audit(new_annotation)
        self.assertEqual(self.system.commit_decision(Ref("A", 2))["support"], "C@2")
        self.assertEqual(self.system.commit_decision(other_decision), {"action": "independent"})
        events = self.system.audit_export()
        self.assertTrue(any(event["kind"] == "verification_result" and event["payload"]["subject"] == Ref("Q", 2).to_dict()
                            and event["payload"]["passed"] for event in events))
        self.assertTrue(any(event["kind"] == "certificate_transition" and event["payload"]["current"] == "SUPERSEDED"
                            for event in events))
        self.assertTrue(AuditJournal.verify(events, self.system.checkpoint()))

    def test_advice_does_not_rescue_a_failed_recovery_obligation(self):
        system = CertificateSystem()
        system.register_verifier("exact-value", exact_value)
        d = system.add_dependency("D", {"value": 4})
        e = system.add_artifact("E", K.EVIDENCE, {"value": 4}, (d,))
        q = system.add_artifact("Q", K.CLAIM, {"value": 4}, (e,))
        c = system.issue_certificate("C", q, "exact-value")
        decision = system.add_artifact("A", K.DECISION, {}, (c,))
        system.register_rebuilder("E", copy_value)
        system.register_rebuilder("Q", lambda previous, supports: {"value": -1})
        system.register_rebuilder("A", lambda previous, supports: {})
        system.execute_recovery(system.plan_recovery(system.change_dependency("D", {"value": 7})))
        adapter = SemanticAdviceAdapter(system)
        prepared = adapter.prepare(system.latest("D").ref, system.latest("Q").ref)
        annotation = adapter.consume(prepared, fixture_response(prepared, html_result(confidence="100.00%")))
        self.assertEqual(annotation["disposition"], "ACCEPTED")
        self.assertEqual(system.certificate_status(c), S.INVALID)
        with self.assertRaisesRegex(ValueError, "Verification failed"):
            system.issue_certificate("C", Ref("Q", 2), "exact-value")
        self.assertEqual(system.latest("C").ref, c)
        self.assertEqual(system.certificate_status(c), S.INVALID)
        self.assertFalse(system.audit_export()[-1]["payload"]["passed"])
        with self.assertRaises(ValueError):
            system.commit_decision(decision)
        self.assertEqual(system.artifact(Ref("Q", 2)).supports, (Ref("E", 2),))
        self.assertTrue(AuditJournal.verify(system.audit_export(), system.checkpoint()))

    def test_old_response_cannot_be_rebound_even_when_text_is_unchanged(self):
        prepared = self.adapter.prepare(self.d, self.q)
        response = fixture_response(prepared)
        self.system.execute_recovery(self.system.plan_recovery(self.system.change_dependency("D", {"value": 4})))
        rebound = replace(prepared, bindings=tuple((node.ref, node.content_hash) for node in
                          (self.system.latest("D"), self.system.latest("Q"))))
        self.assertEqual(rebound.input_json, prepared.input_json)
        self.assertNotEqual(rebound.request_id, prepared.request_id)
        annotation = self.adapter.consume(rebound, response)
        self.assertEqual(annotation["disposition"], "IGNORED")
        self.assertEqual(annotation["reason"], "response_request_mismatch")
        self.assertIsNone(annotation["nli_relation"])
        self.assert_audit(annotation)

    def test_response_normalization_or_provenance_tampering_is_ignored(self):
        for field, value, reason in (("normalized_output", {"label": "entailment"}, "response_normalization_mismatch"),
                                    ("verified_loaded_model_id", "claimed-model", "unsupported_model_attestation"),
                                    ("loaded_model_revision", "claimed-revision", "unsupported_model_attestation"),
                                    ("profile", "nli-cpu", "response_profile_mismatch")):
            with self.subTest(field=field):
                prepared = self.adapter.prepare(self.d, self.q)
                response = fixture_response(prepared)
                response[field] = value
                annotation = self.adapter.consume(prepared, response)
                self.assertEqual(annotation["disposition"], "IGNORED")
                self.assertEqual(annotation["reason"], reason)
                self.assert_audit(annotation)
                self.assertEqual(self.system.certificate_status(self.c), S.VALID)

    def test_invalid_record_or_raw_response_is_ignored_and_audited(self):
        for response in (None, {}, {"schema_version": 1}, {"bad": object()}):
            with self.subTest(response_type=type(response).__name__):
                prepared = self.adapter.prepare(self.d, self.q)
                annotation = self.adapter.consume(prepared, response)
                self.assertEqual(annotation["disposition"], "IGNORED")
                self.assertEqual(annotation["reason"], "malformed_response_record")
                self.assertIn("validation_error", annotation)
                self.assert_audit(annotation)
        prepared = self.adapter.prepare(self.d, self.q)
        non_json = fixture_response(prepared)
        non_json["bad"] = object()
        annotation = self.adapter.consume(prepared, non_json)
        self.assertEqual(annotation["disposition"], "IGNORED")
        self.assertEqual(annotation["reason"], "malformed_response_record")
        self.assert_audit(annotation)
        response = fixture_response(prepared)
        response["output"] = "<div>Consistent"
        annotation = self.adapter.consume(prepared, response)
        self.assertEqual(annotation["disposition"], "IGNORED")
        self.assertEqual(annotation["validation_error"]["type"], "ResponseValidationError")
        self.assert_audit(annotation)
        self.system.commit_decision(self.a)

    def test_hash_mismatch_is_stale_and_prepare_rejects_invalid_inputs(self):
        prepared = self.adapter.prepare(self.d, self.q)
        bad_binding = replace(prepared, bindings=((self.d, "0" * 64), prepared.bindings[1]))
        annotation = self.adapter.consume(bad_binding, fixture_response(bad_binding))
        self.assertEqual(annotation["disposition"], "REJECTED_STALE")
        self.assertEqual(annotation["freshness_failures"][0]["reason"], "input_hash_mismatch")
        self.assert_audit(annotation)
        self.system.change_dependency("D", {"value": 5})
        with self.assertRaisesRegex(ValueError, "stale or invalid"):
            self.adapter.prepare(self.d, self.q)

    def test_neutral_and_contradiction_annotations_are_not_dependency_judgments(self):
        for raw, label in ((html_result("Contradiction"), "contradiction"),
                           (html_result("Unrelated", overlap=.2, similarity=.3), "neutral")):
            with self.subTest(label=label):
                prepared = self.adapter.prepare(self.d, self.q)
                annotation = self.adapter.consume(prepared, fixture_response(prepared, raw))
                self.assertEqual(annotation["disposition"], "ACCEPTED")
                self.assertEqual(annotation["nli_relation"], label)
                self.assertIsNone(annotation["dependency_judgment"])
                self.assertEqual(self.system.artifact(self.q).supports, (self.e,))
                self.assertEqual(self.system.certificate_status(self.c), S.VALID)
                self.assert_audit(annotation)

    def test_network_guard_is_active(self):
        with socket.socket() as connection:
            with self.assertRaisesRegex(AssertionError, "prohibit network"):
                connection.connect(("127.0.0.1", 9))


if __name__ == "__main__":
    unittest.main()
