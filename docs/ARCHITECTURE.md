# Architecture and API

`CertificateSystem` owns the versioned dependency graph, certificate store,
verifier/rebuilder registry and hash-chained audit journal. Support edges point
from an exact artifact ID/version to a consumer. The public package imports no
model libraries and performs no network requests.

## Build and recover

```python
from cert_recovery import ArtifactKind, CertificateSystem, VerificationResult

system = CertificateSystem()

def matches_source(subject, ancestry):
    sources = [node for node in ancestry if node.kind == ArtifactKind.DEPENDENCY]
    passed = bool(sources) and all(
        node.payload["value"] == subject.payload["value"] for node in sources
    )
    return VerificationResult(passed, "claim must equal its declared source")

def rebuild_value(previous, current_supports):
    return {"value": current_supports[0].payload["value"]}

system.register_verifier("matches-source", matches_source)
source = system.add_dependency("source", {"value": 4})
claim = system.add_artifact("claim", ArtifactKind.CLAIM, {"value": 4}, (source,))
system.register_rebuilder("claim", rebuild_value)
certificate = system.issue_certificate("certificate", claim, "matches-source")
decision = system.add_artifact(
    "decision", ArtifactKind.DECISION, {"action": "mock-accept"}, (certificate,)
)
system.register_rebuilder(
    "decision", lambda previous, supports: {"action": "mock-accept"}
)
system.commit_decision(decision)

impact = system.change_dependency("source", {"value": 9})
assert not system.is_valid(certificate)
outcome = system.execute_recovery(system.plan_recovery(impact))
assert outcome.status == "COMPLETE"
assert system.latest("claim").payload == {"value": 9}
system.commit_decision(system.latest("decision").ref)
```

This arithmetic rule is a usage example, not a general-purpose verifier. A failed
registered verifier, missing recipe, untrusted support, revoked authority or
insufficient recovery budget blocks affected work.

## Freshness and certificate lifecycle

Artifacts are immutable JSON snapshots; payload access returns a copy. Supports
bind exact versions. Cycles, missing supports and artifact-ID kind changes are
rejected. Changes consider consumers of all earlier source versions, including
repeated updates before recovery.

Successful recovery moves the old certificate through
`VALID -> INVALID -> PENDING_REVERIFICATION -> SUPERSEDED`; a newly issued
certificate becomes `VALID`. Revoked and superseded certificates are terminal.
Recovery rebuilds derived work rather than copying old claims. Independent valid
branches remain usable. `commit_decision` checks the entire current ancestry under
the engine lock and records only a mock action. Epoch checks reject stale plans
and delayed, explicitly fenced verification requests.

## Optional semantic annotations

The repository-root `semantic_advice.py` adapter consumes schema-2 client records;
it never submits inference. `prepare` captures ordered artifact IDs, versions,
complete hashes and text, with a request identity binding them. `consume` checks
correlation/provenance and reparses the raw response through the reviewed profile.

The engine checks integrity, current versions and valid ancestry under its lock
before recording `ACCEPTED`, `IGNORED` or `REJECTED_STALE` with reasons. Acceptance
is an annotation at consumption time, not permanent freshness. Consume again
before subsequent use; never rebind old advice to new artifacts.

NLI labels remain separate from dependency judgments. Advice does not delete
trusted edges, alter verifier outcomes, issue certificates, mutate recovery
plans or authorize decisions. A failed deterministic obligation remains failed;
advice failure does not add an obligation to an otherwise sufficient workflow.

## Integrity and trust

SHA-256 binds artifact contents and certificate dependency snapshots. Audit-chain
checks detect ordered edits/reordering; truncation requires an independently
retained length/head checkpoint. Control of both journal and checkpoint permits
rewriting history. Canonicalization is project JSON, not full RFC 8785.

Dependency capture, registered rules and the process are trusted. There is no
signature implementation, Byzantine tolerance, durable storage or distributed
atomicity. Direct graph/store mutation bypasses the facade. Graph closure is
conservative only relative to declared edges; missing edges are not discovered
by hashing or semantic advice.
