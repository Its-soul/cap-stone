"""Architectural leakage and oracle-import boundary tests.

Verifies:
1. Runtime modules do not import the isolated scoring oracle (`cert_recovery.oracle`)
   or evaluation harnesses (`ablation`, `composition_evaluation`, `experiments`,
   `model_evaluation`, `evaluation`) via static AST inspection and fresh subprocess
   `sys.modules` checks.
2. Runtime modules contain zero references to evaluation labels or oracle expectations
   (distinguishing legitimate runtime access to policy/claim facts via `rule_decision`).
3. Stripping or mutating oracle labels in evaluation fixtures produces identical
   runtime pipeline outputs (ignoring timestamps and timestamp-derived IDs/hashes).
"""
from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

from cert_recovery.adjudication.composition import assess_specialist, compose_specialist_assessments
from cert_recovery.adjudication.consensus import assess
from cert_recovery.adjudication.domain import FAULT_MODES, PROMPTS, ROLES, InsuranceClaim, RuleAgent
from cert_recovery.adjudication.model_agent import ModelAgentAdapter
from cert_recovery.adjudication.model_evaluation import mock_model_client
from cert_recovery.adjudication.workflow import InsuranceAdjudication, SpecialistCompositionAdjudication
from cert_recovery.oracle.scoring import (
    compliance_evaluation_cases,
    extended_scenarios,
    historical_cases,
    mixed_scenarios,
    mutate_oracle_labels,
    strip_oracle_labels,
)

pytestmark = pytest.mark.adjudication

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_MODULE_PATHS = [
    REPO_ROOT / "src/cert_recovery/engine.py",
    REPO_ROOT / "src/cert_recovery/certificates.py",
    REPO_ROOT / "src/cert_recovery/graph.py",
    REPO_ROOT / "src/cert_recovery/models.py",
    REPO_ROOT / "src/cert_recovery/provenance.py",
    REPO_ROOT / "src/cert_recovery/crypto.py",
    REPO_ROOT / "src/cert_recovery/interfaces.py",
    REPO_ROOT / "src/cert_recovery/v2_inference.py",
    REPO_ROOT / "src/cert_recovery/adjudication/__init__.py",
    REPO_ROOT / "src/cert_recovery/adjudication/domain.py",
    REPO_ROOT / "src/cert_recovery/adjudication/consensus.py",
    REPO_ROOT / "src/cert_recovery/adjudication/composition.py",
    REPO_ROOT / "src/cert_recovery/adjudication/independent_verifier.py",
    REPO_ROOT / "src/cert_recovery/adjudication/ledger.py",
    REPO_ROOT / "src/cert_recovery/adjudication/workflow.py",
    REPO_ROOT / "src/cert_recovery/adjudication/model_agent.py",
    REPO_ROOT / "semantic_advice.py",
]

FORBIDDEN_ORACLE_IMPORTS = {
    "oracle",
    "scoring",
    "ablation",
    "composition_evaluation",
    "experiments",
    "model_evaluation",
    "evaluation",
}


def _imported_module_parts(source: str) -> set[str]:
    tree = ast.parse(source)
    parts: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts.update(alias.name.split("."))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                parts.update(node.module.split("."))
            for alias in node.names:
                parts.add(alias.name)
    return parts


def test_runtime_modules_never_import_scoring_oracle_ast():
    """Fail if any runtime module statically imports the scoring oracle or eval harnesses."""
    for path in RUNTIME_MODULE_PATHS:
        assert path.exists(), f"Expected runtime module missing: {path}"
        source = path.read_text(encoding="utf-8")
        # Allow lazy __getattr__ inside adjudication/__init__.py only for backward compatibility;
        # inspect top-level module imports for __init__.py and full AST for all other runtime modules.
        if path.name == "__init__.py" and path.parent.name == "adjudication":
            tree = ast.parse(source)
            top_level_imports: set[str] = set()
            for stmt in tree.body:
                if isinstance(stmt, ast.Import):
                    for alias in stmt.names:
                        top_level_imports.update(alias.name.split("."))
                elif isinstance(stmt, ast.ImportFrom):
                    if stmt.module:
                        top_level_imports.update(stmt.module.split("."))
                    for alias in stmt.names:
                        top_level_imports.add(alias.name)
            forbidden = top_level_imports & FORBIDDEN_ORACLE_IMPORTS
        else:
            forbidden = _imported_module_parts(source) & FORBIDDEN_ORACLE_IMPORTS
        assert not forbidden, f"Runtime module {path} imports forbidden oracle/eval modules: {sorted(forbidden)}"


def test_runtime_imports_leave_oracle_unloaded_in_sys_modules():
    """Verify importing runtime modules in a clean interpreter never loads oracle modules."""
    code = (
        "import sys; "
        "import cert_recovery.engine; "
        "import cert_recovery.adjudication.domain; "
        "import cert_recovery.adjudication.consensus; "
        "import cert_recovery.adjudication.composition; "
        "import cert_recovery.adjudication.independent_verifier; "
        "import cert_recovery.adjudication.ledger; "
        "import cert_recovery.adjudication.workflow; "
        "import cert_recovery.adjudication.model_agent; "
        "loaded = [m for m in sys.modules if 'oracle' in m or m.endswith('.ablation') or m.endswith('.composition_evaluation') or m.endswith('.model_evaluation') or m.endswith('.experiments')]; "
        "assert not loaded, f'Oracle/eval modules loaded at runtime: {loaded}'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Subprocess failed:\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}"


def test_runtime_adjudication_modules_contain_no_oracle_label_references():
    """Static leakage scan across adjudication runtime modules (excluding legitimate fact rules)."""
    forbidden_pattern = re.compile(
        r"\b(ground_truth|expected_label|expected_claim_label|expected_role_decision|oracle_required)\b",
        re.IGNORECASE,
    )
    runtime_adj = [
        REPO_ROOT / "src/cert_recovery/adjudication/domain.py",
        REPO_ROOT / "src/cert_recovery/adjudication/consensus.py",
        REPO_ROOT / "src/cert_recovery/adjudication/composition.py",
        REPO_ROOT / "src/cert_recovery/adjudication/independent_verifier.py",
        REPO_ROOT / "src/cert_recovery/adjudication/ledger.py",
        REPO_ROOT / "src/cert_recovery/adjudication/workflow.py",
        REPO_ROOT / "src/cert_recovery/adjudication/model_agent.py",
        REPO_ROOT / "src/cert_recovery/engine.py",
    ]
    for path in runtime_adj:
        text = path.read_text(encoding="utf-8")
        matches = forbidden_pattern.findall(text)
        assert not matches, f"Found oracle/label reference in runtime module {path}: {matches}"


_EPHEMERAL_KEYS = frozenset({
    "timestamp",
    "created_at",
    "content_hash",
    "event_hash",
    "payload_hash",
    "dependency_hash",
    "entry_hash",
    "previous_hash",
    "provenance_reference",
    "parents",
    "dag_checkpoint",
    "audit_checkpoint",
})


def _scrub_timestamps_and_hashes(obj):
    if isinstance(obj, dict):
        return {
            k: _scrub_timestamps_and_hashes(v)
            for k, v in obj.items()
            if k not in _EPHEMERAL_KEYS
        }
    if isinstance(obj, list):
        return [_scrub_timestamps_and_hashes(item) for item in obj]
    return obj


def _normalize_runtime_export(data: dict) -> dict:
    """Strip timestamps and timestamp-coupled hashes/references for invariance comparison."""
    return _scrub_timestamps_and_hashes(json.loads(json.dumps(data)))


@pytest.mark.parametrize("mode", ["composition", "legacy"])
def test_runtime_pipeline_invariant_to_stripped_and_mutated_oracle_labels(mode):
    """Runtime workflow outputs remain identical when evaluation oracle labels are stripped or mutated."""
    raw_suite = extended_scenarios() + historical_cases() + mixed_scenarios()
    stripped_suite = strip_oracle_labels(raw_suite)
    mutated_reject = mutate_oracle_labels(raw_suite, target_label="REJECT")
    mutated_approve = mutate_oracle_labels(raw_suite, target_label="APPROVE")
    mutated_garbage = mutate_oracle_labels(raw_suite, target_label="CORRUPTED_ORACLE")

    fault_schedules = [
        {},
        {"risk_agent": "failure"},
        {"policy_agent": "inflated_confidence", "evidence_agent": "malformed"},
    ]

    for idx in range(len(raw_suite)):
        for faults in fault_schedules:
            claim_orig = raw_suite[idx]["claim"]
            claim_stripped = stripped_suite[idx]["claim"]
            claim_mut1 = mutated_reject[idx]["claim"]
            claim_mut2 = mutated_approve[idx]["claim"]
            claim_mut3 = mutated_garbage[idx]["claim"]

            baseline = _normalize_runtime_export(
                InsuranceAdjudication(claim_orig, faults=faults, mode=mode).export()
            )
            for variant_claim in (claim_stripped, claim_mut1, claim_mut2, claim_mut3):
                actual = _normalize_runtime_export(
                    InsuranceAdjudication(variant_claim, faults=faults, mode=mode).export()
                )
                assert actual == baseline


def test_validator_composer_and_adapter_invariant_to_label_mutation():
    """Verify validator, composer, and model adapter ignore stripped/mutated oracle labels."""
    cases = compliance_evaluation_cases()
    stripped = strip_oracle_labels(cases)
    mutated = mutate_oracle_labels(cases, target_label="INVERTED_ORACLE")

    adapter = ModelAgentAdapter(client=mock_model_client)
    rule_agent = RuleAgent()

    for orig_case, strip_case, mut_case in zip(cases, stripped, mutated):
        assert "expected_role_decision" not in strip_case
        assert "expected_claim_label" not in strip_case
        assert mut_case["expected_role_decision"] == "INVERTED_ORACLE"

        outputs = []
        for variant in (orig_case, strip_case, mut_case):
            claim = variant["claim"]
            binding = {"artifact_id": "facts:compliance", "version": 1, "content_hash": "fixed"}
            comp_facts = claim.facts("compliance")
            comp_raw = adapter.execute("compliance", PROMPTS["compliance"], comp_facts, binding)
            comp_val = assess(comp_raw, "compliance", comp_facts, binding)
            comp_spec = assess_specialist(comp_raw, "compliance", comp_facts, binding)

            assessments = {"compliance": comp_spec}
            for role in ("policy", "evidence", "risk"):
                f = claim.facts(role)
                b = {"artifact_id": f"facts:{role}", "version": 1, "content_hash": "fixed"}
                raw = rule_agent.execute(role, PROMPTS[role], f, b)
                assessments[role] = assess_specialist(raw, role, f, b)

            composed = compose_specialist_assessments(assessments)
            outputs.append((
                comp_raw["decision"],
                comp_val,
                comp_spec,
                composed.decision,
                composed.gate,
            ))

        assert outputs[0] == outputs[1] == outputs[2]
