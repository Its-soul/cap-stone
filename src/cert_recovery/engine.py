from __future__ import annotations

from dataclasses import asdict
from math import isfinite
from threading import RLock
from typing import Any

from .certificates import CertificateStore
from .crypto import sha256
from .graph import DependencyGraph
from .interfaces import Rebuilder, Verifier
from .models import (Artifact, ArtifactKind, Certificate, CertificateStatus, ImpactReport,
                     RecoveryOutcome, RecoveryPlan, RecoveryStep, Ref)
from .provenance import AuditJournal


class CertificateSystem:
    """Single-process trusted core. All supported mutations use this facade."""

    def __init__(self, analysis_cost_units: float = 0.0) -> None:
        if not isfinite(analysis_cost_units) or analysis_cost_units < 0:
            raise ValueError("Analysis cost must be finite and nonnegative")
        self._graph = DependencyGraph()
        self._certificates = CertificateStore()
        self._journal = AuditJournal()
        self._invalid: set[Ref] = set()
        self._verifiers: dict[str, tuple[str, Verifier]] = {}
        self._recipes: dict[str, Rebuilder] = {}
        self._lock = RLock()
        self._epoch = 0
        self.analysis_cost_units = analysis_cost_units

    @property
    def epoch(self) -> int:
        return self._epoch

    def artifact(self, ref: Ref) -> Artifact:
        return self._graph.get(ref)

    def latest(self, artifact_id: str) -> Artifact:
        return self._graph.latest(artifact_id)

    def current_artifacts(self) -> tuple[Artifact, ...]:
        return self._graph.current_nodes()

    def certificate(self, ref: Ref) -> Certificate:
        return self._certificates.get(ref)

    def certificate_status(self, ref: Ref) -> CertificateStatus:
        return self._certificates.status(ref)

    def certificate_history(self, ref: Ref | None = None):
        return self._certificates.history(ref)

    def audit_export(self) -> list[dict]:
        with self._lock:
            return self._journal.export()

    def checkpoint(self):
        with self._lock:
            return self._journal.checkpoint()

    def provenance(self, ref: Ref) -> tuple[Artifact, ...]:
        with self._lock:
            refs = self._graph.ancestors(ref) | {ref}
            return tuple(self._graph.get(item) for item in self._graph.topological_order(refs))

    def register_verifier(self, method: str, verifier: Verifier, version: str = "1") -> None:
        with self._lock:
            if not method or not version or method in self._verifiers or not callable(verifier):
                raise ValueError("Verifier names are nonempty and immutable once registered")
            self._verifiers[method] = (version, verifier)
            self._journal.append("verifier_registered", {"method": method, "version": version})

    def register_rebuilder(self, artifact_id: str, recipe: Rebuilder) -> None:
        with self._lock:
            if not artifact_id or artifact_id in self._recipes or not callable(recipe):
                raise ValueError("Duplicate or empty rebuilder ID")
            self._recipes[artifact_id] = recipe
            self._journal.append("rebuilder_registered", {"artifact_id": artifact_id})

    def _append_node(self, artifact_id: str, kind: ArtifactKind, payload: Any,
                     supports: tuple[Ref, ...], source: str, cost: float) -> Artifact:
        try:
            version = self._graph.latest_ref(artifact_id).version + 1
        except KeyError:
            version = 1
        node = Artifact.create(Ref(artifact_id, version), kind, payload, tuple(supports), source, cost)
        self._graph.add(node)
        self._journal.append("artifact_created", {**node.body(), "content_hash": node.content_hash})
        return node

    def _assert_new_id(self, artifact_id: str) -> None:
        if not artifact_id or artifact_id.startswith("verification:"):
            raise ValueError("Empty or reserved artifact ID")
        try:
            self._graph.latest_ref(artifact_id)
        except KeyError:
            return
        raise ValueError("ID exists; use dependency change or recovery to create a new version")

    def add_dependency(self, artifact_id: str, payload: Any, source: str = "local",
                       trusted: bool = True) -> Ref:
        with self._lock:
            self._assert_new_id(artifact_id)
            node = self._append_node(artifact_id, ArtifactKind.DEPENDENCY, payload, (), source, 0.0)
            if not trusted:
                self._invalid.add(node.ref)
                self._journal.append("dependency_untrusted", {"ref": node.ref.to_dict()})
            self._epoch += 1
            return node.ref

    def add_artifact(self, artifact_id: str, kind: ArtifactKind, payload: Any,
                     supports: tuple[Ref, ...], cost_units: float = 1.0, source: str = "local") -> Ref:
        with self._lock:
            self._assert_new_id(artifact_id)
            if kind not in {ArtifactKind.EVIDENCE, ArtifactKind.CLAIM, ArtifactKind.DECISION} or not supports:
                raise ValueError("Derived artifacts require evidence/claim/decision kind and supports")
            if kind == ArtifactKind.DECISION and any(
                self._graph.get(ref).kind != ArtifactKind.CERTIFICATE for ref in supports
            ):
                raise ValueError("Decisions must be supported by certificates")
            if not all(self._is_valid(ref) for ref in supports):
                raise ValueError("Cannot derive from stale or invalid support")
            node = self._append_node(artifact_id, kind, payload, supports, source, cost_units)
            self._epoch += 1
            return node.ref

    def _is_valid(self, ref: Ref) -> bool:
        refs = self._graph.ancestors(ref) | {ref}
        for item in refs:
            node = self._graph.get(item)
            if (item in self._invalid or self._graph.latest_ref(item.artifact_id) != item
                    or not node.integrity_valid()):
                return False
            if node.kind == ArtifactKind.CERTIFICATE:
                if self._certificates.status(item) != CertificateStatus.VALID:
                    return False
                certificate = self._certificates.get(item)
                registered = self._verifiers.get(certificate.method)
                if not registered or registered[0] != certificate.verifier_version:
                    return False
                if certificate.dependency_hash != self._snapshot_hash(certificate.snapshot):
                    return False
        return True

    def is_valid(self, ref: Ref) -> bool:
        with self._lock:
            return self._is_valid(ref)

    def _snapshot_hash(self, snapshot: tuple[Ref, ...]) -> str:
        return sha256([{**ref.to_dict(), "content_hash": self._graph.get(ref).content_hash}
                       for ref in snapshot])

    def _transition(self, ref: Ref, status: CertificateStatus, reason: str) -> None:
        event = self._certificates.transition(ref, status, reason)
        self._journal.append("certificate_transition", asdict(event))

    def _issue_certificate(self, artifact_id: str, subject: Ref, method: str,
                           cost_units: float) -> Ref:
        if not artifact_id or artifact_id.startswith("verification:"):
            raise ValueError("Empty or reserved certificate ID")
        if not self._is_valid(subject):
            raise ValueError("Subject or its support is stale/invalid")
        if self._graph.get(subject).kind not in {ArtifactKind.DEPENDENCY, ArtifactKind.EVIDENCE, ArtifactKind.CLAIM}:
            raise ValueError("Certificate subjects must be dependencies, evidence or claims")
        if not isfinite(cost_units) or cost_units < 0:
            raise ValueError("Certificate cost must be finite and nonnegative")
        parent = None
        try:
            parent = self._graph.latest_ref(artifact_id)
        except KeyError:
            pass
        if parent is not None:
            if self._graph.get(parent).kind != ArtifactKind.CERTIFICATE:
                raise ValueError("Certificate ID collides with another artifact")
            if self._certificates.status(parent) in {CertificateStatus.REVOKED, CertificateStatus.SUPERSEDED}:
                raise ValueError("Terminal certificate cannot be automatically reissued")
            if parent in self._graph.ancestors(subject):
                raise ValueError("Certificate cannot depend on its own prior version")
        version, verifier = self._verifiers[method]
        snapshot = tuple(sorted(self._graph.ancestors(subject) | {subject}))
        result = verifier(self._graph.get(subject), tuple(self._graph.get(ref) for ref in snapshot))
        self._journal.append("verification_result", {"subject": subject.to_dict(), "method": method,
                                                      "verifier_version": version, **result.to_dict()})
        if not result.passed:
            raise ValueError(f"Verification failed: {result.reason}")
        receipt = self._append_node(f"verification:{artifact_id}", ArtifactKind.VERIFICATION,
                                    {"method": method, "version": version, **result.to_dict()},
                                    (subject,), "trusted-verifier", 0.0)
        payload = {"subject": subject.to_dict(), "verification": receipt.ref.to_dict(),
                   "method": method, "verifier_version": version,
                   "snapshot": [ref.to_dict() for ref in snapshot],
                   "dependency_hash": self._snapshot_hash(snapshot),
                   "parent_certificate": parent.to_dict() if parent else None, "issued_epoch": self._epoch}
        node = self._append_node(artifact_id, ArtifactKind.CERTIFICATE, payload,
                                 (subject, receipt.ref), "trusted-verifier", cost_units)
        certificate = Certificate(node.ref, subject, receipt.ref, method, version, snapshot,
                                  payload["dependency_hash"], parent, self._epoch)
        self._certificates.add(certificate)
        self._journal.append("certificate_issued", {"ref": node.ref.to_dict(), "status": "VALID",
                                                   "content_hash": node.content_hash})
        if parent is not None:
            if self._certificates.status(parent) == CertificateStatus.INVALID:
                self._transition(parent, CertificateStatus.PENDING_REVERIFICATION, "replacement verified")
            self._transition(parent, CertificateStatus.SUPERSEDED, f"replaced by {node.ref}")
            self._invalidate({parent}, "certificate superseded")
        return node.ref

    def issue_certificate(self, artifact_id: str, subject: Ref, method: str,
                          cost_units: float = 3.0, expected_epoch: int | None = None) -> Ref:
        with self._lock:
            if expected_epoch is not None and expected_epoch != self._epoch:
                raise ValueError("Stale verification request epoch")
            ref = self._issue_certificate(artifact_id, subject, method, cost_units)
            self._epoch += 1
            return ref

    def _invalidate(self, roots: set[Ref], reason: str) -> ImpactReport:
        affected = set(roots)
        for root in roots:
            affected.update(self._graph.descendants(root))
        self._invalid.update(affected)
        certificates = {ref for ref in affected if self._graph.get(ref).kind == ArtifactKind.CERTIFICATE
                        and self._graph.latest_ref(ref.artifact_id) == ref}
        direct = {ref for ref in certificates if ref in roots or set(self._graph.get(ref).supports) & roots}
        for ref in sorted(certificates):
            if self._certificates.status(ref) in {CertificateStatus.VALID, CertificateStatus.PENDING_REVERIFICATION}:
                self._transition(ref, CertificateStatus.INVALID, reason)
        unaffected = tuple(ref for ref in self._certificates.refs()
                           if ref not in affected and self._graph.latest_ref(ref.artifact_id) == ref
                           and self._is_valid(ref))
        decisions = tuple(sorted(ref for ref in affected if self._graph.get(ref).kind == ArtifactKind.DECISION
                                 and self._graph.latest_ref(ref.artifact_id) == ref))
        report = ImpactReport(self._epoch, tuple(sorted(roots)), tuple(sorted(affected)), tuple(sorted(direct)),
                              tuple(sorted(certificates - direct)), unaffected, decisions)
        self._journal.append("invalidation", {"reason": reason, **asdict(report)})
        return report

    def change_dependency(self, artifact_id: str, payload: Any, trusted: bool = True,
                          reason: str = "dependency changed", source: str | None = None) -> ImpactReport:
        with self._lock:
            old = self._graph.latest(artifact_id)
            if old.kind != ArtifactKind.DEPENDENCY:
                raise ValueError("Only dependencies can be externally changed")
            node = self._append_node(artifact_id, old.kind, payload, (), source or old.source, old.cost_units)
            if not trusted:
                self._invalid.add(node.ref)
                self._journal.append("dependency_untrusted", {"ref": node.ref.to_dict()})
            self._epoch += 1
            previous_versions = {item.ref for item in self._graph.all_nodes()
                                 if item.ref.artifact_id == artifact_id and item.ref != node.ref}
            return self._invalidate(previous_versions, reason)

    def remove_dependency(self, artifact_id: str) -> ImpactReport:
        return self.change_dependency(artifact_id, {"removed": True}, trusted=False, reason="dependency removed")

    def revoke_certificate(self, ref: Ref, reason: str) -> ImpactReport:
        with self._lock:
            self._transition(ref, CertificateStatus.REVOKED, reason)
            self._epoch += 1
            return self._invalidate({ref}, reason)

    def invalidate_artifact(self, ref: Ref, reason: str) -> ImpactReport:
        with self._lock:
            if not reason:
                raise ValueError("Invalidation requires a reason")
            self._graph.get(ref)
            self._epoch += 1
            return self._invalidate({ref}, reason)

    def plan_recovery(self, report: ImpactReport, budget_units: float | None = None) -> RecoveryPlan:
        with self._lock:
            if report.epoch != self._epoch:
                raise ValueError("Impact report is stale")
            if budget_units is not None and (not isfinite(budget_units) or budget_units < 0):
                raise ValueError("Budget must be finite and nonnegative")
            current = {ref for ref in report.affected if self._graph.latest_ref(ref.artifact_id) == ref}
            priorities = {ref: sum(self._graph.get(item).kind == ArtifactKind.DECISION
                                  and self._graph.latest_ref(item.artifact_id) == item
                                  for item in self._graph.descendants(ref)) for ref in current}
            order = self._graph.topological_order(current, lambda ref: (-priorities[ref],
                                                                       self._graph.get(ref).cost_units, str(ref)))
            steps = tuple(RecoveryStep(ref, self._graph.get(ref).kind, self._graph.get(ref).cost_units,
                                       priorities[ref]) for ref in order
                          if self._graph.get(ref).kind not in {ArtifactKind.DEPENDENCY, ArtifactKind.VERIFICATION})
            plan = RecoveryPlan(self._epoch, steps, budget_units, self.analysis_cost_units)
            self._journal.append("recovery_planned", asdict(plan))
            return plan

    def plan_pending_recovery(self, budget_units: float | None = None) -> RecoveryPlan:
        with self._lock:
            pending = tuple(sorted(node.ref for node in self._graph.current_nodes()
                                   if not self._is_valid(node.ref)))
            report = ImpactReport(self._epoch, (), pending, (), (), (), ())
            return self.plan_recovery(report, budget_units)

    def execute_recovery(self, plan: RecoveryPlan) -> RecoveryOutcome:
        with self._lock:
            if plan.epoch != self._epoch:
                raise ValueError("Recovery plan is stale")
            if (plan.analysis_cost_units != self.analysis_cost_units
                    or len({step.ref for step in plan.steps}) != len(plan.steps)
                    or (plan.budget_units is not None and
                        (not isfinite(plan.budget_units) or plan.budget_units < 0))):
                raise ValueError("Invalid recovery plan configuration")
            for step in plan.steps:
                node = self._graph.get(step.ref)
                if (step.ref != self._graph.latest_ref(step.ref.artifact_id)
                        or step.kind != node.kind or step.cost_units != node.cost_units
                        or step.kind in {ArtifactKind.DEPENDENCY, ArtifactKind.VERIFICATION}):
                    raise ValueError("Recovery step does not match current artifact")
            restored: list[Ref] = []
            attempted: list[Ref] = []
            blocked: list[tuple[str, str]] = []
            spent = plan.analysis_cost_units
            if plan.budget_units is not None and spent > plan.budget_units:
                outcome = RecoveryOutcome((), (("analysis", "budget insufficient"),), 0.0, "BLOCKED")
                self._journal.append("recovery_finished", asdict(outcome))
                return outcome
            for step in plan.steps:
                old = self._graph.get(step.ref)
                if self._is_valid(step.ref):
                    continue
                if plan.budget_units is not None and spent + step.cost_units > plan.budget_units:
                    blocked.append((old.ref.artifact_id, "budget insufficient"))
                    continue
                if old.kind == ArtifactKind.CERTIFICATE:
                    if self._certificates.status(old.ref) == CertificateStatus.REVOKED:
                        blocked.append((old.ref.artifact_id, "revoked certificate requires explicit new authority"))
                        continue
                    certificate = self._certificates.get(old.ref)
                    subject = self._graph.latest_ref(certificate.subject.artifact_id)
                    if not self._is_valid(subject):
                        blocked.append((old.ref.artifact_id, "subject/support remains invalid"))
                        continue
                    spent += step.cost_units
                    attempted.append(old.ref)
                    self._transition(old.ref, CertificateStatus.PENDING_REVERIFICATION, "recovery started")
                    try:
                        ref = self._issue_certificate(old.ref.artifact_id, subject, certificate.method, old.cost_units)
                    except Exception as error:
                        self._transition(old.ref, CertificateStatus.INVALID, "re-verification failed")
                        blocked.append((old.ref.artifact_id, str(error)))
                        continue
                else:
                    recipe = self._recipes.get(old.ref.artifact_id)
                    supports = tuple(self._graph.latest_ref(ref.artifact_id) for ref in old.supports)
                    if not recipe or not all(self._is_valid(ref) for ref in supports):
                        blocked.append((old.ref.artifact_id, "missing rebuilder or invalid support"))
                        continue
                    spent += step.cost_units
                    attempted.append(old.ref)
                    try:
                        payload = recipe(old, tuple(self._graph.get(ref) for ref in supports))
                        ref = self._append_node(old.ref.artifact_id, old.kind, payload, supports,
                                                old.source, old.cost_units).ref
                    except Exception as error:
                        blocked.append((old.ref.artifact_id, str(error)))
                        continue
                restored.append(ref)
            self._epoch += 1
            outcome = RecoveryOutcome(tuple(restored), tuple(blocked), spent,
                                      "COMPLETE" if not blocked else "PARTIAL" if restored else "BLOCKED", tuple(attempted))
            self._journal.append("recovery_finished", asdict(outcome))
            return outcome

    def commit_decision(self, ref: Ref) -> dict:
        # Freshness check and mock commit share the lock; no external side effect exists in this phase.
        with self._lock:
            node = self._graph.get(ref)
            if node.kind != ArtifactKind.DECISION or not self._is_valid(ref):
                raise ValueError("Decision cannot commit with stale/invalid certificates")
            self._journal.append("decision_committed", {"ref": ref.to_dict(), "epoch": self._epoch,
                                                        "content_hash": node.content_hash})
            return node.payload
