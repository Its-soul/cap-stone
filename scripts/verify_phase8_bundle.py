#!/usr/bin/env python3
"""Phase 8 verification gate for manuscript, citations, provenance, and numeric parity."""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_TITLE = (
    "Verifiable Multi-Agent Fault-Tolerant Adjudication System "
    "with Cryptographic Causal Provenance"
)

FORBIDDEN_PATTERNS = [
    ("candidate_v3", re.compile(r"Candidate\s+V3", re.IGNORECASE)),
    ("nvidia_multi_epoch", re.compile(r"NVIDIA\s+multi-epoch", re.IGNORECASE)),
    ("todo_marker", re.compile(r"\bTODO\b")),
    ("fill_in_marker", re.compile(r"\[FILL\s*IN\]", re.IGNORECASE)),
    ("unresolved_qmark", re.compile(r"\?\?")),
    ("placeholder_author", re.compile(r"\b(?:Anonymous|Your Name|Department of X)\b", re.IGNORECASE)),
    ("secret_groq_key", re.compile(r"\bgsk_[A-Za-z0-9]{12,}")),
    ("private_key_block", re.compile(r"BEGIN\s+(?:OPENSSH|RSA|EC|PRIVATE)\s+KEY")),
]


def _load_json(rel_path: str) -> dict[str, Any]:
    return json.loads((ROOT / rel_path).read_text(encoding="utf-8"))


def verify_phase8_bundle() -> dict[str, Any]:
    """Run all static, bibliographic, privacy, and numeric-parity checks."""
    tex_path = ROOT / "manuscript" / "main.tex"
    bib_path = ROOT / "manuscript" / "references.bib"
    audit_path = ROOT / "results" / "phase8" / "citation_audit.md"
    claim_map_path = ROOT / "results" / "phase8" / "claim_to_artifact_map.md"
    manifest_path = ROOT / "results" / "phase8" / "reproducibility_manifest.json"

    for required in (tex_path, bib_path, audit_path, claim_map_path, manifest_path):
        if not required.exists():
            raise FileNotFoundError(f"Missing Phase 8 deliverable: {required}")

    tex = tex_path.read_text(encoding="utf-8")
    bib = bib_path.read_text(encoding="utf-8")
    audit_md = audit_path.read_text(encoding="utf-8")

    # 1. Exact title check
    title_match = re.search(r"\\title\{([^}]+)\}", tex)
    actual_title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else ""
    if actual_title != EXPECTED_TITLE:
        raise AssertionError(f"Title mismatch: {actual_title!r} != {EXPECTED_TITLE!r}")

    # 2. Forbidden tokens / secret leak scan across manuscript + phase8 docs
    scanned_texts = {
        "manuscript/main.tex": tex,
        "manuscript/references.bib": bib,
        "results/phase8/citation_audit.md": audit_md,
        "results/phase8/claim_to_artifact_map.md": claim_map_path.read_text(encoding="utf-8"),
        "results/phase8/reproducibility_manifest.json": manifest_path.read_text(encoding="utf-8"),
    }
    for name, body in scanned_texts.items():
        for label, pat in FORBIDDEN_PATTERNS:
            if pat.search(body):
                raise AssertionError(f"Forbidden pattern {label!r} found in {name}")

    # 3. Citation completeness & zero orphans
    cite_keys: set[str] = set()
    for match in re.finditer(r"\\cite[pt]?\{([^}]+)\}", tex):
        for raw_key in match.group(1).split(","):
            key = raw_key.strip()
            if key:
                cite_keys.add(key)

    bib_keys = set(re.findall(r"@\w+\{([^,\s]+)\s*,", bib))
    missing_cites = sorted(cite_keys - bib_keys)
    orphan_bib = sorted(bib_keys - cite_keys)
    if missing_cites:
        raise AssertionError(f"Missing BibTeX entries: {missing_cites}")
    if orphan_bib:
        raise AssertionError(f"Orphan BibTeX entries not cited in main.tex: {orphan_bib}")

    # Ensure every bib key is also audited in citation_audit.md
    unaudited = sorted(k for k in bib_keys if f"`{k}`" not in audit_md)
    if unaudited:
        raise AssertionError(f"BibTeX keys missing from citation_audit.md: {unaudited}")

    # 4. Internal \ref -> \label resolution
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", tex))
    unresolved_refs = sorted(refs - labels)
    if unresolved_refs:
        raise AssertionError(f"Unresolved LaTeX \\ref labels: {unresolved_refs}")

    # 5. Numeric parity checks against Phase 2-7 artifacts
    p2 = _load_json("results/phase2/stale_certificate_replay_matrix.json")
    p3 = _load_json("results/phase3/provenance_tamper_matrix.json")
    p3_causal = _load_json("results/phase3/causal_crosscheck.json")
    p4_pilot = _load_json("results/phase4/pilot_summary.json")
    p4_faults = _load_json("results/phase4/fault_matrix.json")
    p6_tables = _load_json("results/phase6/tables_and_metrics.json")
    p6_attempt = _load_json("results/phase6/attempt_manifest.json")
    p7 = _load_json("results/phase7/v2_appendix_summary.json")

    assert p2["passed_scenarios"] == 7 and p2["total_scenarios"] == 7
    assert p3["summary"]["tamper_attacks_detected"] == 9
    assert p3["summary"]["tamper_attacks_attempted"] == 9
    assert p3["summary"]["tamper_attacks_accepted"] == 0
    assert p3_causal["post_evidence_update_v1_to_v2"]["data_lineage_matches_artifact_dag"] is True
    assert p4_pilot["live_execution"]["budget"]["calls_used"] == 58
    assert p4_pilot["live_execution"]["budget"]["total_tokens"] == 26829
    assert len(p4_faults["controlled_injected_faults"]) == 8

    clean = p6_tables["offline_clean_N104"]["arms"]
    live = p6_tables["live_arms"]
    sweep = p6_tables["offline_10seed_sweep_N104x10_exploratory"]["arms"]
    assert clean["FULL_DET"]["correct_all"] == 104
    assert clean["FULL_DET"]["unsafe_approvals"] == 0
    assert clean["MAJORITY_POLICY_VETO"]["correct_all"] == 77
    assert clean["MAJORITY_POLICY_VETO"]["unsafe_approvals"] == 27
    assert clean["MAJORITY_VOTE"]["correct_all"] == 31
    assert clean["MAJORITY_VOTE"]["unsafe_approvals"] == 70
    assert clean["CONFIDENCE_WEIGHTED_VOTE"]["correct_all"] == 28
    assert clean["CONFIDENCE_WEIGHTED_VOTE"]["unsafe_approvals"] == 73
    assert sweep["FULL_DET"]["correct_all"] == 733
    assert sweep["FULL_DET"]["unsafe_approvals"] == 0
    assert live["FULL_LIVE"]["metrics"]["correct_all"] == 36
    assert live["LLM_AS_JUDGE"]["metrics"]["correct_all"] == 30
    assert live["SINGLE_AGENT_E2E_LLM"]["metrics"]["correct_all"] == 28
    assert p6_attempt["live_track"]["budget"]["calls_used"] == 325
    assert p6_attempt["live_track"]["budget"]["total_tokens"] == 171888

    # Verify exact Phase 6 + Phase 7 key strings appear verbatim in manuscript
    required_tex_snippets = [
        "0.7404", "0.3375", "1.490\\times 10^{-8}",
        "0.2981", "0.8750", "0.2692", "0.9125",
        "0.7048", "0.6019", "0.5075",
        "171{,}888", "74{,}886", "23{,}524", "17{,}117",
        "0.8333", "0.7778",
        "35 / 128", "12 / 48", "23.210", "21.490", "0.721", "0.8125",
        "0.8750 ($56/64$)", "0.9167 ($88/96$)", "0.8281", "0.8464",
    ]
    missing_snippets = [s for s in required_tex_snippets if s not in tex]
    if missing_snippets:
        raise AssertionError(f"Expected empirical figures missing from main.tex: {missing_snippets}")

    cm7 = p7["confusion_matrices"]
    assert cm7["final_candidate_v2"]["accuracy"] == 88 / 96
    assert cm7["final_original"]["accuracy"] == 0.5
    assert cm7["stress_candidate_v2"]["fp"] == 66
    assert cm7["stress_original"]["fp"] == 59

    # 6. Check reproducibility manifest paths exist & live outside _private/
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, rel in manifest["verified_artifact_sources"].items():
        if rel.startswith("_private"):
            raise AssertionError(f"Manifest source {key} points inside _private/: {rel}")
        if not (ROOT / rel).exists():
            raise AssertionError(f"Manifest source {key} does not exist: {rel}")

    compilers = {
        name: shutil.which(name)
        for name in ("pdflatex", "xelatex", "lualatex", "tectonic", "bibtex")
    }
    pdf_compile_status = (
        "AVAILABLE"
        if any(compilers.get(c) for c in ("pdflatex", "xelatex", "lualatex", "tectonic"))
        else "BLOCKED (no local LaTeX toolchain)"
    )

    return {
        "status": "PASS",
        "title": actual_title,
        "citations_count": len(cite_keys),
        "bib_entries_count": len(bib_keys),
        "labels_count": len(labels),
        "refs_count": len(refs),
        "compilers": compilers,
        "pdf_compile_status": pdf_compile_status,
    }


def main() -> int:
    report = verify_phase8_bundle()
    out_path = ROOT / "results" / "phase8" / "verify_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
