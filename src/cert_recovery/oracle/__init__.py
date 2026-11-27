"""Isolated scoring-oracle and ground-truth evaluation fixtures.

This package contains benchmark labels, pure-oracle evaluators (Arm F), and
metric calculators used strictly for offline evaluation. Runtime adjudication
modules (domain, validator, composer, ledger, workflow, engine, model_agent)
must never import this package.
"""

from .scoring import (
    SCORING_ORACLE_MODULES,
    calculate_metrics,
    calculate_paired_disagreements,
    compliance_evaluation_cases,
    evaluate_rule_oracle,
    extended_scenarios,
    historical_cases,
    legacy_experiment_cases,
    mixed_scenarios,
    mutate_oracle_labels,
    strip_oracle_labels,
)

__all__ = [
    "SCORING_ORACLE_MODULES",
    "calculate_metrics",
    "calculate_paired_disagreements",
    "compliance_evaluation_cases",
    "evaluate_rule_oracle",
    "extended_scenarios",
    "historical_cases",
    "legacy_experiment_cases",
    "mixed_scenarios",
    "mutate_oracle_labels",
    "strip_oracle_labels",
]
