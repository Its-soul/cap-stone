"""Offline insurance research prototype; no model or distributed BFT claims."""

from .composition import (
    COMPOSITION_CONTRACT_VERSION,
    CompositionResult,
    SpecialistAssessment,
    assess_specialist,
    compose_specialist_assessments,
)
from .workflow import InsuranceAdjudication, SpecialistCompositionAdjudication

__all__ = [
    "InsuranceAdjudication",
    "SpecialistCompositionAdjudication",
    "SpecialistAssessment",
    "CompositionResult",
    "assess_specialist",
    "compose_specialist_assessments",
    "COMPOSITION_CONTRACT_VERSION",
    "run_ablation_study",
]


def __getattr__(name: str):
    """Lazily expose evaluation helpers without importing oracle/ablation into runtime."""
    if name == "run_ablation_study":
        from .ablation import run_ablation_study
        return run_ablation_study
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
