import copy
import unittest
from dataclasses import replace

from cert_recovery import ArtifactKind as K, CertificateStatus as S, CertificateSystem, VerificationResult
from cert_recovery.provenance import AuditJournal


def copy_value(previous, supports):
    return {"value": supports[0].payload["value"]}


def exact_value(subject, ancestry):
    roots = [node for node in ancestry if node.kind == K.DEPENDENCY]
    passed = bool(roots) and all(node.payload.get("value") == subject.payload.get("value") for node in roots)
    return VerificationResult(passed, "subject equals declared source value" if passed else "source mismatch")


def make_system():
    system = CertificateSystem()
    system.register_verifier("exact-value", exact_value)
    d = system.add_dependency("D", {"value": 4})
    e = system.add_artifact("E", K.EVIDENCE, {"value": 4}, (d,))
    q = system.add_artifact("Q", K.CLAIM, {"value": 4}, (e,))
    c = system.issue_certificate("C", q, "exact-value")
    decision = system.add_artifact("A", K.DECISION, {"action": "mock-accept"}, (c,))
    system.register_rebuilder("E", copy_value)
    system.register_rebuilder("Q", copy_value)
    system.register_rebuilder("A", lambda previous, supports: {"action": "mock-accept", "support": str(supports[0].ref)})
    return system, d, e, q, c, decision


class EngineTests(unittest.TestCase):
    def test_chain_and_provenance(self):
        s, d, e, q, c, a = make_system()
        self.assertEqual(s.commit_decision(a), {"action": "mock-accept"})
        self.assertEqual({node.ref for node in s.provenance(a)}, {d, e, q, c, a, s.certificate(c).verification})
        self.assertIsNone(s.artifact(s.certificate(c).verification).payload["confidence"])

    def test_direct_and_transitive_impact_and_unaffected_branch(self):
        s, d, e, q, c, a = make_system()
        direct = s.issue_certificate("Direct", d, "exact-value")
        u = s.add_dependency("U", {"value": 99})
        unaffected = s.issue_certificate("Other", u, "exact-value")
        report = s.change_dependency("D", {"value": 5})
        self.assertIn(direct, report.direct_certificates)
        self.assertIn(c, report.transitive_certificates)
        self.assertIn(unaffected, report.unaffected_certificates)
        self.assertEqual(report.decisions, (a,))
        self.assertFalse(s.is_valid(q))
        self.assertTrue(s.is_valid(unaffected))
        self.assertEqual(s.certificate_status(c), S.INVALID)
        with self.assertRaises(ValueError):
            s.commit_decision(a)

    def test_recovery_rebuilds_values_and_preserves_history(self):
        s, d, e, q, c, a = make_system()
        original_hash = s.artifact(q).content_hash
        report = s.change_dependency("D", {"value": 9})
        plan = s.plan_recovery(report)
        self.assertLess([x.ref.artifact_id for x in plan.steps].index("Q"),
                        [x.ref.artifact_id for x in plan.steps].index("C"))
        outcome = s.execute_recovery(plan)
        self.assertEqual(outcome.status, "COMPLETE")
        self.assertEqual(s.latest("Q").payload["value"], 9)
        self.assertTrue(s.is_valid(s.latest("C").ref))
        self.assertEqual(s.artifact(q).content_hash, original_hash)
        self.assertEqual(s.artifact(q).payload["value"], 4)
        self.assertEqual(s.certificate_status(c), S.SUPERSEDED)
        self.assertEqual([event.current for event in s.certificate_history(c)],
                         [S.VALID, S.INVALID, S.PENDING_REVERIFICATION, S.SUPERSEDED])
        self.assertEqual(s.certificate(s.latest("C").ref).parent_certificate, c)
        self.assertEqual(s.commit_decision(s.latest("A").ref)["support"], "C@2")
        self.assertFalse(s.is_valid(c))
        self.assertTrue(AuditJournal.verify(s.audit_export(), s.checkpoint()))

    def test_failed_verification_does_not_issue_a_certificate(self):
        s = CertificateSystem()
        s.register_verifier("reject", lambda subject, ancestry: VerificationResult(False, "policy rejects"))
        d = s.add_dependency("D", {})
        with self.assertRaisesRegex(ValueError, "Verification failed"):
            s.issue_certificate("C", d, "reject")
        with self.assertRaises(KeyError):
            s.latest("C")

    def test_failed_reverification_remains_invalid(self):
        s, d, e, q, c, a = make_system()
        s._recipes["Q"] = lambda previous, supports: {"value": -1}
        outcome = s.execute_recovery(s.plan_recovery(s.change_dependency("D", {"value": 7})))
        self.assertEqual(outcome.status, "PARTIAL")
        self.assertEqual(s.certificate_status(c), S.INVALID)
        self.assertFalse(s.is_valid(a))
        self.assertIn("Verification failed", dict(outcome.blocked)["C"])

    def test_missing_recipe_blocks_without_copying_old_claim(self):
        s, d, e, q, c, a = make_system()
        del s._recipes["E"]
        outcome = s.execute_recovery(s.plan_recovery(s.change_dependency("D", {"value": 7})))
        self.assertEqual(outcome.status, "BLOCKED")
        self.assertEqual(s.latest("Q").ref, q)
        self.assertEqual(s.certificate_status(c), S.INVALID)
        s.register_rebuilder("E", copy_value)
        self.assertEqual(s.execute_recovery(s.plan_pending_recovery()).status, "COMPLETE")

    def test_budget_cannot_restore_unverified_decision(self):
        s, d, e, q, c, a = make_system()
        report = s.change_dependency("D", {"value": 7})
        outcome = s.execute_recovery(s.plan_recovery(report, budget_units=2))
        self.assertLessEqual(outcome.attempted_cost_units, 2)
        self.assertFalse(s.is_valid(c))
        with self.assertRaises(ValueError):
            s.commit_decision(a)
        self.assertEqual(s.execute_recovery(s.plan_pending_recovery()).status, "COMPLETE")

    def test_untrusted_and_removed_dependencies_block_recovery(self):
        for change in (lambda s: s.change_dependency("D", {"value": 7}, trusted=False),
                       lambda s: s.remove_dependency("D")):
            s, d, e, q, c, a = make_system()
            outcome = s.execute_recovery(s.plan_recovery(change(s)))
            self.assertEqual(outcome.status, "BLOCKED")
            self.assertFalse(s.is_valid(s.latest("D").ref))
            self.assertEqual(s.artifact(d).payload, {"value": 4})

    def test_source_failure_then_replacement_can_recover(self):
        s, d, e, q, c, a = make_system()
        s.execute_recovery(s.plan_recovery(s.remove_dependency("D")))
        s.change_dependency("D", {"value": 8}, trusted=True)
        self.assertEqual(s.execute_recovery(s.plan_pending_recovery()).status, "COMPLETE")
        self.assertEqual(s.latest("Q").payload["value"], 8)

    def test_stale_plan_and_verification_request_rejected(self):
        s, d, e, q, c, a = make_system()
        report = s.change_dependency("D", {"value": 7})
        plan = s.plan_recovery(report)
        s.change_dependency("D", {"value": 8})
        with self.assertRaises(ValueError):
            s.execute_recovery(plan)
        with self.assertRaises(ValueError):
            s.plan_recovery(report)
        with self.assertRaises(ValueError):
            s.issue_certificate("Delayed", s.latest("D").ref, "exact-value", expected_epoch=plan.epoch)

    def test_repeated_changes_before_recovery_capture_old_bindings(self):
        s, d, e, q, c, a = make_system()
        s.change_dependency("D", {"value": 7})
        report = s.change_dependency("D", {"value": 8})
        self.assertIn(c, report.certificates)
        self.assertEqual(s.execute_recovery(s.plan_recovery(report)).status, "COMPLETE")
        self.assertEqual(s.latest("Q").payload, {"value": 8})

    def test_certificate_to_certificate_cascade(self):
        s, d, e, q, c, a = make_system()
        chain_claim = s.add_artifact("ChainClaim", K.CLAIM, {"value": 4}, (c,))
        chain_cert = s.issue_certificate("ChainCert", chain_claim, "exact-value")
        chain_decision = s.add_artifact("ChainDecision", K.DECISION, {}, (chain_cert,))
        s.register_rebuilder("ChainClaim", lambda previous, supports: s.latest("Q").payload)
        s.register_rebuilder("ChainDecision", lambda previous, supports: {"support": str(supports[0].ref)})
        report = s.change_dependency("D", {"value": 11})
        self.assertIn(chain_cert, report.transitive_certificates)
        self.assertIn(chain_decision, report.decisions)
        self.assertEqual(s.execute_recovery(s.plan_recovery(report)).status, "COMPLETE")
        self.assertTrue(s.is_valid(s.latest("ChainDecision").ref))
        self.assertEqual(s.latest("ChainClaim").payload["value"], 11)

    def test_plan_cannot_forge_cost_or_analysis_budget(self):
        s, *_ = make_system()
        plan = s.plan_recovery(s.change_dependency("D", {"value": 5}))
        with self.assertRaises(ValueError):
            s.execute_recovery(replace(plan, steps=(replace(plan.steps[0], cost_units=0),)))
        with self.assertRaises(ValueError):
            s.execute_recovery(replace(plan, analysis_cost_units=99))

    def test_disputed_intermediate_claim_is_rebuilt(self):
        s, d, e, q, c, a = make_system()
        report = s.invalidate_artifact(q, "external checker disputes intermediate claim")
        self.assertIn(c, report.direct_certificates)
        self.assertTrue(s.is_valid(e))
        self.assertEqual(s.execute_recovery(s.plan_recovery(report)).status, "COMPLETE")
        self.assertEqual(s.latest("Q").ref.version, 2)
        self.assertTrue(s.is_valid(s.latest("A").ref))

    def test_analysis_budget_failure_is_audited(self):
        s, *_ = make_system()
        s.analysis_cost_units = 1
        plan = s.plan_recovery(s.change_dependency("D", {"value": 5}), budget_units=0)
        outcome = s.execute_recovery(plan)
        self.assertEqual(outcome.status, "BLOCKED")
        self.assertEqual(s.audit_export()[-1]["kind"], "recovery_finished")

    def test_revocation_is_not_automatic_recovery(self):
        s, d, e, q, c, a = make_system()
        report = s.revoke_certificate(c, "authority withdrawn")
        outcome = s.execute_recovery(s.plan_recovery(report))
        self.assertEqual(s.certificate_status(c), S.REVOKED)
        self.assertEqual(outcome.status, "BLOCKED")
        with self.assertRaises(ValueError):
            s.issue_certificate("C", q, "exact-value")

    def test_payload_access_returns_copy(self):
        s, d, e, q, c, a = make_system()
        value = s.artifact(d).payload
        value["value"] = 123
        self.assertEqual(s.artifact(d).payload["value"], 4)

    def test_replacement_certificate_invalidates_downstream(self):
        s, d, e, q, c, a = make_system()
        new = s.issue_certificate("C", q, "exact-value")
        self.assertTrue(s.is_valid(new))
        self.assertFalse(s.is_valid(a))
        self.assertEqual(s.certificate_status(c), S.SUPERSEDED)

    def test_invalid_support_and_id_collision_rejected(self):
        s, d, e, q, c, a = make_system()
        with self.assertRaises(ValueError):
            s.issue_certificate("Q", q, "exact-value")
        s.change_dependency("D", {"value": 7})
        with self.assertRaises(ValueError):
            s.add_artifact("Bad", K.CLAIM, {}, (q,))

    def test_audit_edit_reorder_and_truncation(self):
        s, *_ = make_system()
        original = s.audit_export()
        checkpoint = s.checkpoint()
        altered = copy.deepcopy(original)
        altered[0]["payload"]["method"] = "changed"
        self.assertFalse(AuditJournal.verify(altered, checkpoint))
        reordered = copy.deepcopy(original)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        self.assertFalse(AuditJournal.verify(reordered, checkpoint))
        self.assertTrue(AuditJournal.verify(original[:-1]))
        self.assertFalse(AuditJournal.verify(original[:-1], checkpoint))
        self.assertEqual(s.audit_export(), original)
