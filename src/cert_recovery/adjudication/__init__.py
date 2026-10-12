"""Offline insurance research prototype; no model or distributed BFT claims."""

from .ablation import run_ablation_study
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
