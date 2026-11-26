# Architecture and Data Contracts

`CertificateSystem` (`src/cert_recovery/engine.py`) owns the versioned dependency graph, certificate store, verifier/rebuilder registry, and SHA-256 hash-chained audit journal (`EventDAG`). Support edges point from an exact artifact ID/version to a consumer. The core package imports no ML/model libraries and performs no network requests.

## Hybrid Architecture Boundary

The system enforces a strict boundary between untrusted advisory ML models and authoritative deterministic verification:

- **Untrusted Advisory ML Layer:** Candidate V2 dependency prediction (`DeBERTa-v3-small`), semantic NLI checks (`MiniLM2`, `DeBERTa-v3-small`), and generative specialist proposers (`Qwen2.5-0.5B-Instruct`). ML outputs have **zero decisional or certificate authority**; they cannot issue certificates, create authoritative DAG edges, or override mandatory rules.
- **Authoritative Deterministic Layer:** `CertificateSystem`, `EventDAG`, `MandatoryRuleGate`, typed schema validation, and `SpecialistCompositionAdjudication`. Any failed, malformed, or stale specialist output falls back safely to human `REVIEW` rather than `APPROVE`.

## Core Certificate Engine and Selective Recovery

Artifacts are immutable JSON snapshots; payload access returns a copy. Supports bind exact `(artifact_id, version)` pairs, and cycles, missing supports, or kind changes are rejected.

```python
from cert_recovery import ArtifactKind, CertificateSystem, VerificationResult

system = CertificateSystem()

def matches_source(subject, ancestry):
    sources = [node for node in ancestry if node.kind == ArtifactKind.DEPENDENCY]
    passed = bool(sources) and all(
        node.payload["value"] == subject.payload["value"] for node in sources
    )
    return VerificationResult(passed, "claim must equal its declared source")

system.register_verifier("matches-source", matches_source)
source = system.add_dependency("source", {"value": 4})
claim = system.add_artifact("claim", ArtifactKind.CLAIM, {"value": 4}, (source,))
system.register_rebuilder("claim", lambda prev, sup: {"value": sup[0].payload["value"]})
certificate = system.issue_certificate("certificate", claim, "matches-source")
decision = system.add_artifact("decision", ArtifactKind.DECISION, {"action": "mock-accept"}, (certificate,))
system.register_rebuilder("decision", lambda prev, sup: {"action": "mock-accept"})
system.commit_decision(decision)

impact = system.change_dependency("source", {"value": 9})
assert not system.is_valid(certificate)
outcome = system.execute_recovery(system.plan_recovery(impact))
assert outcome.status == "COMPLETE"
assert system.latest("claim").payload == {"value": 9}
```

Successful recovery transitions prior certificates through `VALID -> INVALID -> PENDING_REVERIFICATION -> SUPERSEDED`, issuing a new `VALID` certificate for rebuilt artifacts while leaving independent valid branches untouched.

### Integrity and Trust Model

- **Hash Chaining:** SHA-256 binds artifact payloads, certificate dependency snapshots, and `EventDAG` parent links.
- **Tamper Detection:** Audit-chain checks detect payload edits and reordering; detecting history truncation requires an independently retained external checkpoint `(length, head_hash)`.
- **Limits:** Canonicalization uses deterministic project JSON (not full RFC 8785). There are no asymmetric digital signatures, distributed transactions, or Byzantine consensus protocols; control of both the event store and external anchor permits rewriting history.

## Specialist-Role Composition Contract (`composition-v1`)

In the legacy consensus prototype, non-policy specialists (`evidence`, `risk`, `compliance`) voted `APPROVE` whenever their local check passed, creating a 3-to-1 majority against a valid `policy` rejection and inducing 42 false `REVIEW` outcomes across 45 rejection trials. `SpecialistCompositionAdjudication` (`src/cert_recovery/adjudication/composition.py`) resolves this category error by assigning disjoint mandatory domain obligations instead of interchangeable whole-claim votes:

| Role | Contract | Evaluated Predicates | Finding if Met | Finding if Unmet | Permitted Consequence | Impossible Consequence |
|---|---|---|---|---|---|---|
| **`policy`** | `coverage-v1` | `hospitalization_covered`, `waiting_period_satisfied`, `amount_inr <= coverage_limit_inr` | `SATISFIED` | `FAILED` | `APPROVE` / `REJECT` | `REVIEW` (as domain outcome) |
| **`evidence`** | `documents-v1` | `documents_available`, `invoice_matches` | `SATISFIED` | `FAILED` | `APPROVE` / `REVIEW` | `REJECT` |
| **`risk`** | `fraud-screen-v1` | `not duplicate_claim`, `not provider_flagged` | `SATISFIED` | `FAILED` | `APPROVE` / `REVIEW` | `REJECT` |
| **`compliance`** | `identity-consent-v1` | `identity_verified`, `consent_available` | `SATISFIED` | `FAILED` | `APPROVE` / `REVIEW` | `REJECT` |

Outputs asserting an impossible consequence are rejected as `IMPOSSIBLE_CONSEQUENCE` and mapped to `UNCERTAIN` (`REVIEW`).

### Truth Table and Precedence Modes

- **`SUBSTANTIVE_EXCLUSION` (Default):**
  1. If `policy` is `UNCERTAIN` (crash, malformed, stale) -> **`REVIEW`**.
  2. If `policy` is `VALID` and `FAILED` -> **`REJECT`** (policy ineligibility overrides procedural checks).
  3. If `policy` is `SATISFIED` and any non-policy specialist is `FAILED` or `UNCERTAIN` -> **`REVIEW`**.
  4. If all four specialists are `VALID` and `SATISFIED` -> **`APPROVE`**.
- **`STRICT_AUDIT` (Alternative):** If any specialist is `UNCERTAIN`, return **`REVIEW`** before evaluating exclusions.

Confidence scores remain diagnostic metadata only and cannot override rules or issue certificates.

## Data Formats and Split Contracts

- **Smoke Fixtures (`baseline_tests.json`):** Six authored directional NLI pairs (`entailment`, `contradiction`, `neutral`), including contrast case `nli-003`.
- **Examples (`data/examples/`):** Format reference JSONL schemas (`dependency_pairs.example.jsonl`, `workflow_cases.example.jsonl`).
- **Dependency Pair Schema:** JSONL records binding `pair_id`, `world_id`, `premise`, `hypothesis`, `label` (`0` independent, `1` depends_on), and `source`.
- **Leakage Controls:** Cleaning normalizes Unicode/whitespace and rejects duplicates across splits. Splits partition by group (`template_family` or `world_id`) using structural projection and hash manifests before model loading.
- **Controlled Pilot & Candidate V2:** 96-row arithmetic pilot (`64/16/16` split, 12 template families) and 416-row Candidate V2 controlled suite (`256/64/96` split plus 384 stress rows) with intervention witnesses verified through `CertificateSystem`. See [METHODOLOGY.md](METHODOLOGY.md) for annotation scope and V3 contrast formulation.
