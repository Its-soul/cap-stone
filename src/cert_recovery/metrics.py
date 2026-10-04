from __future__ import annotations

import math
from dataclasses import dataclass

from .models import ArtifactKind, RecoveryPlan


def ratio(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or numerator < 0 or denominator < 0:
        raise ValueError("Ratio inputs must be finite and nonnegative")
    return numerator / denominator if denominator else None


def recovery_savings(selective_cost: float, full_cost: float) -> float | None:
    fraction = ratio(selective_cost, full_cost)
    return None if fraction is None else 1 - fraction


@dataclass(frozen=True)
class CostComparison:
    selective_units: float
    full_units: float
    savings: float | None
    interpretation: str = "Configured deterministic work units; not measured latency or money"


def compare_plan_cost(plan: RecoveryPlan, current_artifacts) -> CostComparison:
    full = sum(node.cost_units for node in current_artifacts
               if node.kind not in {ArtifactKind.DEPENDENCY, ArtifactKind.VERIFICATION})
    selective = plan.estimated_cost_units
    return CostComparison(selective, full, recovery_savings(selective, full))


def binary_metrics(labels: list[int], predictions: list[int]) -> dict:
    if len(labels) != len(predictions) or any(type(x) is not int or x not in (0, 1) for x in labels + predictions):
        raise ValueError("Aligned binary labels and predictions required")
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    return {"accuracy": ratio(sum(y == p for y, p in zip(labels, predictions)), len(labels)),
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "f1": ratio(2 * tp, 2 * tp + fp + fn), "support": len(labels),
            "tp": tp, "fp": fp, "fn": fn}


def calibration_metrics(labels: list[int], probabilities: list[float], bins: int = 10) -> dict:
    if len(labels) != len(probabilities) or type(bins) is not int or bins < 1:
        raise ValueError("Aligned labels/probabilities and positive bin count required")
    if any(type(x) is not int or x not in (0, 1) for x in labels):
        raise ValueError("Binary labels required")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Probabilities must be finite and in [0,1]")
    if not labels:
        return {"brier": None, "ece": None}
    ece = 0.0
    for index in range(bins):
        members = [i for i, p in enumerate(probabilities) if min(int(p * bins), bins - 1) == index]
        if members:
            observed = sum(labels[i] for i in members) / len(members)
            predicted = sum(probabilities[i] for i in members) / len(members)
            ece += len(members) / len(labels) * abs(observed - predicted)
    return {"brier": sum((p - y) ** 2 for y, p in zip(labels, probabilities)) / len(labels), "ece": ece}


def reliability_summary(safe_commits: int, total_commits: int, completed_tasks: int, total_tasks: int) -> dict:
    if any(type(x) is not int or x < 0 for x in (safe_commits, total_commits, completed_tasks, total_tasks)):
        raise ValueError("Counts must be nonnegative integers")
    if safe_commits > total_commits or completed_tasks > total_tasks:
        raise ValueError("Success counts cannot exceed their denominators")
    return {"support_valid_commit_rate": ratio(safe_commits, total_commits),
            "useful_completion_rate": ratio(completed_tasks, total_tasks)}

