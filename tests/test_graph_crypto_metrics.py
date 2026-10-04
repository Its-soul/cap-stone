import unittest
from dataclasses import replace

from cert_recovery.certificates import CertificateStore
from cert_recovery.crypto import canonical_json, sha256
from cert_recovery.graph import DependencyGraph
from cert_recovery.metrics import (binary_metrics, calibration_metrics, compare_plan_cost,
                                   recovery_savings, reliability_summary)
from cert_recovery.models import (Artifact, ArtifactKind as K, Certificate, CertificateStatus as S,
                                  RecoveryPlan, RecoveryStep, Ref)


class GraphTests(unittest.TestCase):
    def test_diamond_closure_versions_and_order(self):
        g = DependencyGraph()
        d = Artifact.create(Ref("D", 1), K.DEPENDENCY, {}, ())
        g.add(d)
        a = Artifact.create(Ref("A", 1), K.EVIDENCE, {}, (d.ref,))
        b = Artifact.create(Ref("B", 1), K.CLAIM, {}, (d.ref,))
        g.add(a)
        g.add(b)
        c = Artifact.create(Ref("C", 1), K.CERTIFICATE, {}, (a.ref, b.ref))
        g.add(c)
        self.assertEqual(g.descendants(d.ref), {a.ref, b.ref, c.ref})
        self.assertEqual(g.ancestors(c.ref), {d.ref, a.ref, b.ref})
        self.assertEqual(g.affected_certificates(d.ref), (c.ref,))
        self.assertEqual(g.direct_dependents(d.ref), (a.ref, b.ref))
        self.assertEqual(g.topological_order({d.ref, a.ref, b.ref, c.ref})[-1], c.ref)
        g.add(Artifact.create(Ref("D", 2), K.DEPENDENCY, {"new": True}, ()))
        self.assertEqual(g.latest_ref("D"), Ref("D", 2))
        self.assertEqual(g.get(d.ref).payload, {})

    def test_missing_support_duplicate_cycle_and_hash_rejected(self):
        g = DependencyGraph()
        d = Artifact.create(Ref("D", 1), K.DEPENDENCY, {}, ())
        with self.assertRaises(KeyError):
            g.add(Artifact.create(Ref("E", 1), K.EVIDENCE, {}, (d.ref,)))
        g.add(d)
        with self.assertRaises(ValueError):
            g.add(d)
        a = Artifact.create(Ref("A", 1), K.CLAIM, {}, (d.ref,))
        g.add(a)
        with self.assertRaises(ValueError):
            g.add(Artifact.create(Ref("D", 2), K.DEPENDENCY, {}, (a.ref,)))
        with self.assertRaises(ValueError):
            g.add(replace(Artifact.create(Ref("Bad", 1), K.CLAIM, {}, (d.ref,)), content_hash="wrong"))


class IntegrityLifecycleTests(unittest.TestCase):
    def test_canonical_order_and_non_json_rejection(self):
        self.assertEqual(sha256({"a": 1, "b": "हिंदी"}), sha256({"b": "हिंदी", "a": 1}))
        for value in ({1: "key"}, float("nan"), {"s": {1, 2}}, float("inf")):
            with self.assertRaises(ValueError):
                canonical_json(value)

    def test_terminal_lifecycle_and_no_manual_valid_transition(self):
        store = CertificateStore()
        ref = Ref("C", 1)
        store.add(Certificate(ref, Ref("D", 1), Ref("V", 1), "rule", "1", (), "hash", None, 0))
        store.transition(ref, S.INVALID, "change")
        with self.assertRaises(ValueError):
            store.transition(ref, S.VALID, "trust me")
        store.transition(ref, S.PENDING_REVERIFICATION, "try")
        store.transition(ref, S.REVOKED, "policy")
        with self.assertRaises(ValueError):
            store.transition(ref, S.PENDING_REVERIFICATION, "retry")


class CalculationTests(unittest.TestCase):
    def test_savings_zero_and_negative(self):
        self.assertEqual(recovery_savings(3, 10), .7)
        self.assertAlmostEqual(recovery_savings(12, 10), -.2)
        self.assertIsNone(recovery_savings(0, 0))
        with self.assertRaises(ValueError):
            recovery_savings(-1, 10)

    def test_cost_includes_analysis_and_certificate_verification_once(self):
        d = Artifact.create(Ref("D", 1), K.DEPENDENCY, {}, (), cost_units=0)
        c = Artifact.create(Ref("C", 1), K.CERTIFICATE, {}, (d.ref,), cost_units=3)
        other = Artifact.create(Ref("O", 1), K.CLAIM, {}, (d.ref,), cost_units=8)
        plan = RecoveryPlan(1, (RecoveryStep(c.ref, K.CERTIFICATE, 3, 0),), None, 2)
        costs = compare_plan_cost(plan, (d, c, other))
        self.assertEqual(costs.selective_units, 5)
        self.assertEqual(costs.full_units, 11)

    def test_binary_and_undefined_metrics(self):
        metrics = binary_metrics([1, 0, 1, 0], [1, 1, 0, 0])
        self.assertEqual(metrics["precision"], .5)
        self.assertEqual(metrics["recall"], .5)
        self.assertEqual(metrics["f1"], .5)
        self.assertIsNone(binary_metrics([0], [0])["precision"])
        self.assertIsNone(binary_metrics([], [])["accuracy"])
        with self.assertRaises(ValueError):
            binary_metrics([1], [])

    def test_calibration_not_proof_of_truth(self):
        self.assertEqual(calibration_metrics([0, 1], [0., 1.]), {"brier": 0., "ece": 0.})
        self.assertIsNone(calibration_metrics([], [])["ece"])
        with self.assertRaises(ValueError):
            calibration_metrics([0], [float("nan")])

    def test_reliability_does_not_reward_all_abstention(self):
        summary = reliability_summary(0, 0, 0, 10)
        self.assertIsNone(summary["support_valid_commit_rate"])
        self.assertEqual(summary["useful_completion_rate"], 0)
        with self.assertRaises(ValueError):
            reliability_summary(3, 2, 0, 10)
