from __future__ import annotations

from .models import Certificate, CertificateStatus, LifecycleEvent, Ref, utc_now

_TRANSITIONS = {
    CertificateStatus.VALID: {CertificateStatus.INVALID, CertificateStatus.REVOKED, CertificateStatus.SUPERSEDED},
    CertificateStatus.INVALID: {CertificateStatus.PENDING_REVERIFICATION, CertificateStatus.REVOKED},
    CertificateStatus.PENDING_REVERIFICATION: {CertificateStatus.INVALID, CertificateStatus.SUPERSEDED, CertificateStatus.REVOKED},
    CertificateStatus.REVOKED: set(),
    CertificateStatus.SUPERSEDED: set(),
}


class CertificateStore:
    def __init__(self) -> None:
        self._certificates: dict[Ref, Certificate] = {}
        self._states: dict[Ref, CertificateStatus] = {}
        self._history: list[LifecycleEvent] = []

    def get(self, ref: Ref) -> Certificate:
        return self._certificates[ref]

    def refs(self) -> tuple[Ref, ...]:
        return tuple(sorted(self._certificates))

    def status(self, ref: Ref) -> CertificateStatus:
        return self._states[ref]

    def history(self, ref: Ref | None = None) -> tuple[LifecycleEvent, ...]:
        return tuple(event for event in self._history if ref is None or event.ref == ref)

    def add(self, certificate: Certificate) -> LifecycleEvent:
        if certificate.ref in self._certificates:
            raise ValueError("Certificate already exists")
        self._certificates[certificate.ref] = certificate
        self._states[certificate.ref] = CertificateStatus.VALID
        event = LifecycleEvent(certificate.ref, None, CertificateStatus.VALID, "issued", utc_now())
        self._history.append(event)
        return event

    def transition(self, ref: Ref, target: CertificateStatus, reason: str) -> LifecycleEvent:
        current = self.status(ref)
        if not reason or target not in _TRANSITIONS[current]:
            raise ValueError(f"Forbidden lifecycle transition: {current} -> {target}")
        event = LifecycleEvent(ref, current, target, reason, utc_now())
        self._states[ref] = target
        self._history.append(event)
        return event

