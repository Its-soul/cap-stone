from __future__ import annotations

from typing import Protocol

from .models import Artifact, VerificationResult


class Verifier(Protocol):
    """Trusted task rule. Gold benchmark labels must never be passed here."""

    def __call__(self, subject: Artifact, ancestry: tuple[Artifact, ...]) -> VerificationResult: ...


class Rebuilder(Protocol):
    def __call__(self, previous: Artifact, current_supports: tuple[Artifact, ...]) -> dict: ...


class EvidenceProvider(Protocol):
    def fetch(self, query: str) -> dict: ...


class SemanticDependencyAnalyzer(Protocol):
    """Optional dependency-task probabilities; NLI confidence is not this judgment.

    Version-bound NLI annotations use the separate optional semantic_advice
    adapter. Neither interface grants graph or certificate authority.
    """

    def predict_pairs(self, pairs: list[tuple[str, str]]) -> list[float]: ...


class DecisionProposer(Protocol):
    def propose(self, evidence: tuple[Artifact, ...]) -> dict: ...

