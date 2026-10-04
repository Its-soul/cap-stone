from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from math import isfinite
from typing import Any

from .crypto import canonical_json, sha256


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ArtifactKind(StrEnum):
    DEPENDENCY = "dependency"
    EVIDENCE = "evidence"
    CLAIM = "claim"
    VERIFICATION = "verification"
    CERTIFICATE = "certificate"
    DECISION = "decision"


class CertificateStatus(StrEnum):
    VALID = "VALID"
    INVALID = "INVALID"
    PENDING_REVERIFICATION = "PENDING_REVERIFICATION"
    REVOKED = "REVOKED"
    SUPERSEDED = "SUPERSEDED"


@dataclass(frozen=True, order=True)
class Ref:
    artifact_id: str
    version: int

    def __post_init__(self) -> None:
        if not self.artifact_id or type(self.version) is not int or self.version < 1:
            raise ValueError("A reference needs a nonempty ID and positive integer version")

    def to_dict(self) -> dict:
        return {"artifact_id": self.artifact_id, "version": self.version}

    def __str__(self) -> str:
        return f"{self.artifact_id}@{self.version}"


@dataclass(frozen=True)
class Artifact:
    ref: Ref
    kind: ArtifactKind
    payload_json: str
    supports: tuple[Ref, ...]
    created_at: str
    source: str
    cost_units: float
    content_hash: str

    @property
    def payload(self) -> Any:
        return json.loads(self.payload_json)

    def body(self) -> dict:
        return {"ref": self.ref.to_dict(), "kind": self.kind.value, "payload": self.payload,
                "supports": [ref.to_dict() for ref in self.supports],
                "created_at": self.created_at, "source": self.source, "cost_units": self.cost_units}

    def integrity_valid(self) -> bool:
        return self.content_hash == sha256(self.body())

    @classmethod
    def create(cls, ref: Ref, kind: ArtifactKind, payload: Any, supports: tuple[Ref, ...],
               source: str = "local", cost_units: float = 1.0) -> Artifact:
        if not isfinite(cost_units) or cost_units < 0:
            raise ValueError("Cost must be finite and nonnegative")
        body = {"ref": ref.to_dict(), "kind": kind.value, "payload": payload,
                "supports": [item.to_dict() for item in supports],
                "created_at": utc_now(), "source": source, "cost_units": cost_units}
        return cls(ref, kind, canonical_json(payload), supports, body["created_at"], source,
                   cost_units, sha256(body))


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    reason: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        if type(self.passed) is not bool or not self.reason:
            raise ValueError("Verification needs a Boolean result and a reason")
        if self.confidence is not None and (not isfinite(self.confidence) or not 0 <= self.confidence <= 1):
            raise ValueError("Confidence must be in [0, 1]; None means uncalibrated/unknown")

    def to_dict(self) -> dict:
        return {"passed": self.passed, "reason": self.reason, "confidence": self.confidence}


@dataclass(frozen=True)
class Certificate:
    ref: Ref
    subject: Ref
    verification: Ref
    method: str
    verifier_version: str
    snapshot: tuple[Ref, ...]
    dependency_hash: str
    parent_certificate: Ref | None
    issued_epoch: int


@dataclass(frozen=True)
class LifecycleEvent:
    ref: Ref
    previous: CertificateStatus | None
    current: CertificateStatus
    reason: str
    timestamp: str


@dataclass(frozen=True)
class ImpactReport:
    epoch: int
    changed: tuple[Ref, ...]
    affected: tuple[Ref, ...]
    direct_certificates: tuple[Ref, ...]
    transitive_certificates: tuple[Ref, ...]
    unaffected_certificates: tuple[Ref, ...]
    decisions: tuple[Ref, ...]

    @property
    def certificates(self) -> tuple[Ref, ...]:
        return tuple(sorted(set(self.direct_certificates + self.transitive_certificates)))


@dataclass(frozen=True)
class RecoveryStep:
    ref: Ref
    kind: ArtifactKind
    cost_units: float
    downstream_decisions: int


@dataclass(frozen=True)
class RecoveryPlan:
    epoch: int
    steps: tuple[RecoveryStep, ...]
    budget_units: float | None
    analysis_cost_units: float

    @property
    def estimated_cost_units(self) -> float:
        return self.analysis_cost_units + sum(step.cost_units for step in self.steps)


@dataclass(frozen=True)
class RecoveryOutcome:
    restored: tuple[Ref, ...]
    blocked: tuple[tuple[str, str], ...]
    attempted_cost_units: float
    status: str
    attempted: tuple[Ref, ...] = ()
