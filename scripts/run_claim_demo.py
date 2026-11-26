"""Serve a local demo of the existing adjudication workflow."""

import argparse
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from threading import Lock
from urllib.parse import urlsplit
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cert_recovery.adjudication.domain import InsuranceClaim, ROLES, RULES
from cert_recovery.adjudication.workflow import SpecialistCompositionAdjudication
from cert_recovery.v2_inference import V2Predictor


ASSETS = ROOT / "demo"
STATIC = {
    "/": ("index.html", "text/html"),
    "/styles.css": ("styles.css", "text/css"),
    "/app.js": ("app.js", "text/javascript"),
}
FIELD_LABELS = {
    "amount_inr": "Claim amount",
    "coverage_limit_inr": "Policy limit",
    "hospitalization_covered": "Coverage",
    "waiting_period_satisfied": "Waiting period met",
    "documents_available": "Documents available",
    "invoice_matches": "Invoice matches",
    "duplicate_claim": "Duplicate claim",
    "provider_flagged": "Provider flagged",
    "identity_verified": "Identity verified",
    "consent_available": "Consent available",
}
FAILED_FACT_REASONS = {
    "hospitalization_covered": "Hospitalization is outside coverage",
    "waiting_period_satisfied": "Waiting period unmet",
    "documents_available": "Required documents missing",
    "invoice_matches": "Invoice mismatch",
    "duplicate_claim": "Duplicate claim flagged",
    "provider_flagged": "Provider flagged",
    "identity_verified": "Identity unverified",
    "consent_available": "Consent missing",
}
DEFAULT_FACTS = {
    key: value
    for role in ROLES
    for key, value in InsuranceClaim(coverage_limit_inr=65_000).facts(role).items()
}


def form_schema():
    return [
        {"role": role, "fields": [
            {"key": key, "label": FIELD_LABELS[key], "default": DEFAULT_FACTS[key]}
            for key in RULES[role][1]
        ]}
        for role in ROLES
    ]


def decision_summary(workflow):
    result = workflow.result()
    round_ref = workflow.system.certificate(workflow.system.latest("round-certificate").ref).subject
    assessments = workflow.system.artifact(round_ref).payload["specialist_assessments"]
    gate = result["gate"]
    reasons = []
    roles = ("policy",) if gate in {"SUBSTANTIVE_EXCLUSION", "POLICY_UNCERTAIN"} else ROLES
    for role in roles:
        assessment = assessments[role]
        finding = assessment["obligation_result"]
        facts = assessment["evidence_references"]
        if finding == "UNCERTAIN":
            reasons.append(f"{role.title()} check could not be confirmed")
        elif finding == "FAILED":
            reasons.extend(FAILED_FACT_REASONS[key] for key, value in facts.items()
                           if key in FAILED_FACT_REASONS and value != DEFAULT_FACTS[key])
            if role == "policy" and facts["amount_inr"] > facts["coverage_limit_inr"]:
                reasons.append("Claim amount exceeds the policy limit")
    if gate == "ALL_OBLIGATIONS_SATISFIED":
        reasons.append("All checks passed")
    return {"decision": result["decision"], "reason": ". ".join(reasons) or gate.replace("_", " ").capitalize(),
            "gate": gate, "amount_inr": workflow.claim.amount_inr}


def certificate_summary(system):
    ref = system.latest("round-certificate").ref
    return {**ref.to_dict(), "status": system.certificate_status(ref).value}


class ClaimCase:
    def __init__(self, claim):
        self.workflow = SpecialistCompositionAdjudication(claim)
        self.last_decision = decision_summary(self.workflow)
        self.recovery = None

    def submit(self, claim=None):
        workflow = self.workflow
        system = workflow.system
        initial = claim is None
        previous = None if initial else self.last_decision
        certificate_before = None if initial else certificate_summary(system)
        counts_before = dict.fromkeys(ROLES, 0) if initial else workflow.executions.copy()
        events_before = 0 if initial else workflow.ledger.checkpoint()["length"]
        journal_before = 0 if initial else system.checkpoint().length

        changes = []
        if not initial:
            for key in DEFAULT_FACTS:
                old, new = getattr(workflow.claim, key), getattr(claim, key)
                if old != new:
                    changes.append({"field": key, "label": FIELD_LABELS[key], "before": old, "after": new})
            outcome = workflow.update_claim(claim)
            if outcome is not None:
                self.recovery = asdict(outcome)
                if system.is_valid(system.latest("final").ref):
                    self.last_decision = decision_summary(workflow)

        current = self.last_decision if system.is_valid(system.latest("final").ref) else None
        checks = []
        for role in ROLES:
            agent = system.latest(f"{role}_agent")
            valid = system.is_valid(agent.ref)
            ran = workflow.executions[role] > counts_before[role]
            status = ("Ran" if initial else "Reran") if ran else ("Reused" if valid else "Blocked")
            finding = agent.payload["specialist_assessment"]
            checks.append({"role": role, "status": status, "valid": valid,
                           "evidence": finding["evidence_references"], "input": finding["input_binding"],
                           "finding": finding["obligation_result"], "assessment": agent.ref.to_dict()})

        event_types = {"evidence_access", "agent_execution", "specialist_assessment", "specialist_composition",
                       "final_decision", "recovery_started", "recovery"}
        journal_types = {"invalidation", "recovery_planned", "recovery_finished", "certificate_transition",
                         "verification_result", "certificate_issued", "decision_committed"}
        return {
            "claim_id": workflow.claim.claim_id,
            "previous": previous, "current": current, "previous_historical": previous is not None and current is None,
            "changes": changes, "no_changes": not initial and not changes,
            "checks": checks, "recovery": self.recovery if changes or current is None else None,
            "certificate": {"before": certificate_before, "after": certificate_summary(system)},
            "audit": {
                "events": [event for event in workflow.ledger.export()[events_before:]
                           if event["event_type"] in event_types],
                "engine_events": [event for event in system.audit_export()[journal_before:]
                                  if event["kind"] in journal_types],
            },
        }


class DemoServer(ThreadingHTTPServer):
    def __init__(self, address):
        super().__init__(address, DemoHandler)
        self.cases = {}
        self.claim_lock = Lock()
        self.v2 = V2Predictor(ROOT)

    def submit_claim(self, facts, claim_id=None):
        if set(facts) != set(DEFAULT_FACTS):
            raise ValueError("Submit all ten claim facts")
        claim = InsuranceClaim(claim_id=claim_id or uuid4().hex, **facts)
        with self.claim_lock:
            if claim_id is not None:
                return self.cases[claim_id].submit(claim)
            case = ClaimCase(claim)
            response = case.submit()
            self.cases[claim.claim_id] = case
            return response


class DemoHandler(BaseHTTPRequestHandler):
    def send_content(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status, data):
        self.send_content(status, json.dumps(data, allow_nan=False).encode("utf-8"), "application/json")

    def do_GET(self):
        route = urlsplit(self.path).path
        if route == "/api/form":
            self.send_json(HTTPStatus.OK, form_schema())
        elif route in STATIC:
            filename, content_type = STATIC[route]
            self.send_content(HTTPStatus.OK, (ASSETS / filename).read_bytes(), content_type)
        else:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "Page not found"})

    def do_POST(self):
        route = urlsplit(self.path).path
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if self.headers.get_content_type() != "application/json" or not 0 < size <= 65536:
                raise ValueError("Expected a JSON request of at most 64 KB")
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")
            if route == "/api/v2":
                result = self.server.v2.predict(data.get("source"), data.get("assessment"))
            elif route == "/api/claims":
                result = self.server.submit_claim(data)
            elif route.startswith("/api/claims/") and route.rsplit("/", 1)[1]:
                claim_id = route.removeprefix("/api/claims/")
                result = self.server.submit_claim(data, claim_id)
            else:
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint not found"})
                return
        except ValueError as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except KeyError:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "Claim not found. Start a new claim."})
        else:
            self.send_json(HTTPStatus.OK, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    with DemoServer(("127.0.0.1", args.port)) as server:
        print(f"Claim demo at http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
