"""Integrate adjudication with the unchanged certificate/recovery facade."""
import copy
from dataclasses import asdict

from ..crypto import canonical_json
from ..engine import CertificateSystem
from ..models import ArtifactKind as K, VerificationResult
from .consensus import ConfidenceWeightedConsensus, Vote, assess, majority
from .domain import FAULT_MODES, PROMPTS, ROLES, RULES, InsuranceClaim, RuleAgent, rule_decision, simulate_fault
from .ledger import EventDAG


class InsuranceAdjudication:
    """One claim per workflow; trusted local rules, untrusted agent adapters.

    Role votes express local checks. The final safety gate prevents majority
    approval from overriding any mandatory local rule. It is not a learned oracle.
    """

    def __init__(self, claim: InsuranceClaim, *, backends=None, faults=None):
        self.system = CertificateSystem()
        self.ledger = EventDAG()
        self.claim = claim
        self.backends = {role: RuleAgent() for role in ROLES}
        if backends:
            if not set(backends) <= set(ROLES):
                raise ValueError("Unknown backend role")
            self.backends.update(backends)
        self.faults = dict(faults or {})
        if not set(self.faults) <= {r + "_agent" for r in ROLES} or any(m not in FAULT_MODES for m in self.faults.values()):
            raise ValueError("Unknown agent or fault mode")
        self.executions = {role: 0 for role in ROLES}
        self._source_events = {}
        self._recovery_event = None
        self.system.register_verifier("insurance-round-v1", self._verify_round)
        for role in ROLES:
            ref = self.system.add_dependency("facts:" + role, {"claim_id": claim.claim_id, "facts": claim.facts(role),
                    "rule": RULES[role][0], "node_type": "source", "source": "synthetic-insurance-fixture"})
            self._record_source(self.system.artifact(ref))
            agent_id = role + "_agent"
            self.system.register_rebuilder(agent_id, lambda previous, supports, role=role: self._execute(role, supports[0]))
            self.system.add_artifact(agent_id, K.EVIDENCE, self._execute(role, self.system.artifact(ref)), (ref,), source=agent_id)
        self.system.register_rebuilder("round", lambda previous, supports: self._round(supports))
        refs = tuple(self.system.latest(r + "_agent").ref for r in ROLES)
        round_ref = self.system.add_artifact("round", K.CLAIM, self._round(tuple(self.system.artifact(r) for r in refs)), refs)
        certificate = self.system.issue_certificate("round-certificate", round_ref, "insurance-round-v1")
        self.system.register_rebuilder("final", self._rebuild_final)
        self.system.add_artifact("final", K.DECISION, self._final_payload(certificate), (certificate,))

    @staticmethod
    def _binding(artifact):
        return {**artifact.ref.to_dict(), "content_hash": artifact.content_hash}

    def _execute(self, role, source):
        self.executions[role] += 1
        binding = self._binding(source)
        facts = source.payload["facts"]
        source_event = self._record_source(source)
        parents = [source_event] + ([self._recovery_event] if self._recovery_event else [])
        access = self.ledger.append("evidence_access", {"binding": binding, "facts": facts,
                                                     "rule": RULES[role][0]}, parents, role + "_agent")
        execution = self.ledger.append("agent_execution", {"prompt": PROMPTS[role], "input_reference": binding},
                                       (access,), role + "_agent")
        try:
            raw = self.backends[role].execute(role, PROMPTS[role], copy.deepcopy(facts), copy.deepcopy(binding))
            raw = simulate_fault(raw, self.faults.get(role + "_agent"))
            canonical_json(raw)  # Reject non-JSON / NaN before it can enter provenance.
        except Exception as error:
            raw = {"execution_error": type(error).__name__}
        output_event = self.ledger.append("agent_output", raw, (execution,), role + "_agent")
        assessment = assess(raw, role, facts, binding)
        validation = self.ledger.append("validation", asdict(assessment), (output_event, access), role + "_agent")
        fault = self.ledger.append("fault_assessment", asdict(assessment), (validation,), role + "_agent")
        return {"agent_id": role + "_agent", "role": role, "raw_output": raw,
                "assessment": asdict(assessment), "input_reference": binding,
                "provenance_reference": fault, "node_type": "agent-assessment",
                "rule_references": [RULES[role][0]], "evidence": facts,
                "decision_impact": "validated weighted role vote"}

    def _record_source(self, source):
        if source.content_hash not in self._source_events:
            self._source_events[source.content_hash] = self.ledger.append("input_received", {
                "binding": self._binding(source), "payload": source.payload})
        return self._source_events[source.content_hash]

    def _compute_round(self, agents):
        votes = []
        for node in agents:
            data = node.payload
            source = self.system.artifact(node.supports[0])
            assessment = assess(data["raw_output"], data["role"], source.payload["facts"], self._binding(source))
            raw = data["raw_output"]
            votes.append(Vote(data["agent_id"], raw["decision"] if assessment.validation_status != "FAILED" else "REVIEW",
                              raw["confidence"] if assessment.validation_status != "FAILED" else 0.,
                              assessment.validation_score, assessment.fault_score))
        consensus = ConfidenceWeightedConsensus().aggregate(votes)
        local = [rule_decision(r, self.system.latest("facts:" + r).payload["facts"]) for r in ROLES]
        # A verified policy exclusion is authoritative; incomplete documents/risk/controls require review.
        required = "REJECT" if "REJECT" in local else "REVIEW" if "REVIEW" in local else "APPROVE"
        candidate = consensus["decision"]
        # A proposal must pass both weighted consensus and deterministic rules.
        final = candidate if candidate == required else "REVIEW"
        return {"consensus": consensus, "baseline": majority([n.payload["raw_output"] for n in agents]),
                "decision": final, "required_rule_outcome": required,
                "gate": "PASS" if candidate == required else "REVIEW_REQUIRED",
                "votes": [asdict(v) for v in votes], "node_type": "consensus",
                "rule_references": [RULES[r][0] for r in ROLES] + ["mandatory-gate-v1"]}

    def _round(self, agents):
        result = self._compute_round(agents)
        parents = []
        for node, vote in zip(agents, result["votes"]):
            parents.append(self.ledger.append("vote", vote, (node.payload["provenance_reference"],), vote["agent_id"]))
        result["provenance_reference"] = self.ledger.append("consensus", result, parents)
        return result

    def _verify_round(self, subject, ancestry):
        try:
            agents = tuple(self.system.artifact(ref) for ref in subject.supports)
            if [n.ref.artifact_id for n in agents] != [r + "_agent" for r in ROLES]:
                raise ValueError("Round must bind exactly four distinct roles in declared order")
            actual = {k: v for k, v in subject.payload.items() if k != "provenance_reference"}
            expected = self._compute_round(agents)
            passed = actual == expected and self.ledger.verify(self.ledger.export(), self.ledger.checkpoint())
            events = {e["event_hash"]: e for e in self.ledger.export()}
            consensus_event = events[subject.payload["provenance_reference"]]
            passed = passed and consensus_event["event_type"] == "consensus" and consensus_event["payload"] == actual
            vote_events = []
            for role, node, vote in zip(ROLES, agents, expected["votes"]):
                if len(node.supports) != 1 or node.supports[0].artifact_id != "facts:" + role or node.payload["role"] != role:
                    raise ValueError("Role/source binding mismatch")
                source = self.system.artifact(node.supports[0])
                assessment = asdict(assess(node.payload["raw_output"], node.payload["role"], source.payload["facts"], self._binding(source)))
                passed = passed and canonical_json(assessment) == canonical_json(node.payload["assessment"])
                fault_event = events[node.payload["provenance_reference"]]
                passed = passed and fault_event["event_type"] == "fault_assessment" and canonical_json(fault_event["payload"]) == canonical_json(assessment)
                trace = self.ledger.trace(node.payload["provenance_reference"])
                passed = passed and any(e["event_type"] == "agent_output" and e["agent_id"] == role + "_agent"
                                        and e["payload"] == node.payload["raw_output"] for e in trace)
                matches = [e["event_hash"] for e in events.values() if e["event_type"] == "vote"
                           and e["payload"] == vote and e["parents"] == [node.payload["provenance_reference"]]
                           and e["event_hash"] in consensus_event["parents"]]
                if len(matches) != 1:
                    raise ValueError("Vote event binding mismatch")
                vote_events.extend(matches)
            passed = passed and sorted(vote_events) == consensus_event["parents"]
            return VerificationResult(passed, "Recomputed votes, assessments, mandatory gate and event hashes")
        except (KeyError, TypeError, ValueError):
            return VerificationResult(False, "Malformed round")

    def _final_payload(self, certificate):
        cert = self.system.certificate(certificate)
        result = self.system.artifact(cert.subject).payload
        payload = {"decision": result["decision"], "consensus": result["consensus"], "gate": result["gate"],
                   "certificate": self._binding(self.system.artifact(certificate)), "node_type": "final-decision"}
        payload["provenance_reference"] = self.ledger.append("final_decision", payload, (result["provenance_reference"],))
        return payload

    def _rebuild_final(self, previous, supports):
        return self._final_payload(supports[0].ref)

    def result(self):
        return self.system.commit_decision(self.system.latest("final").ref)

    def recover_agent(self, agent_id, *, clear_simulated_fault=True):
        if agent_id not in {r + "_agent" for r in ROLES}:
            raise ValueError("Unknown agent")
        if clear_simulated_fault:
            self.faults.pop(agent_id, None)
        report = self.system.invalidate_artifact(self.system.latest(agent_id).ref, "explicit agent failure recovery")
        return self._recover(report)

    def update_claim(self, claim: InsuranceClaim):
        if claim.claim_id != self.claim.claim_id:
            raise ValueError("Cannot replace claim identity")
        changed = False
        for role in ROLES:
            source = self.system.latest("facts:" + role)
            if source.payload["facts"] != claim.facts(role):
                self.system.change_dependency(source.ref.artifact_id, {**source.payload, "facts": claim.facts(role)})
                changed = True
        self.claim = claim
        return self._recover() if changed else None

    def _recover(self, report=None):
        before = dict(self.executions)
        old_final = self.system.latest("final").payload["provenance_reference"]
        self._recovery_event = self.ledger.append("recovery_started", {"executions": before}, (old_final,))
        plan = self.system.plan_recovery(report) if report else self.system.plan_pending_recovery()
        outcome = self.system.execute_recovery(plan)
        self._recovery_event = None
        self.ledger.append("recovery", {"outcome": asdict(outcome), "executions_before": before,
                "executions_after": self.executions, "audit_checkpoint": asdict(self.system.checkpoint())},
                (old_final, self.system.latest("final").payload["provenance_reference"]))
        return outcome

    def export(self):
        final = self.result()
        return {"synthetic": True, "claim": asdict(self.claim), "final": final,
                "agents": [self.system.latest(r + "_agent").payload for r in ROLES],
                "round": self.system.latest("round").payload, "executions": dict(self.executions),
                "causal_rule_nodes": [{**n.body(), "content_hash": n.content_hash}
                                      for n in self.system.provenance(self.system.latest("final").ref)],
                "events": self.ledger.export(), "dag_checkpoint": self.ledger.checkpoint(),
                "audit_chain": self.system.audit_export(), "audit_checkpoint": asdict(self.system.checkpoint())}
