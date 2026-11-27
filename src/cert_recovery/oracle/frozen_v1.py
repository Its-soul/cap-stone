"""Frozen Phase 5/6 synthetic dataset generator, independent spec-v1 oracle, and blind loader.

Strictly an offline oracle/evaluation module inside `cert_recovery.oracle`.
Runtime adjudication modules MUST NEVER import this module.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import random
from typing import Any

from ..adjudication.domain import ROLES, InsuranceClaim
from ..crypto import canonical_json, sha256
from .scoring import evaluate_rule_oracle

SPEC_VERSION = "spec-v1"
GENERATOR_VERSION = "frozen-gen-v1"
GENERATOR_SEED = 20261127
SWEEP_SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909, 1010)


def derive_spec_v1_labels(claim_dict: dict[str, Any]) -> dict[str, Any]:
    """Derive obligation statuses and final labels directly from `docs/FROZEN_DECISION_SPEC_V1.md`.

    Implemented independently from `RuleAgent`, `composition.py`, and `evaluate_rule_oracle`.
    """
    amt = claim_dict["amount_inr"]
    lim = claim_dict["coverage_limit_inr"]
    if type(amt) is not int or amt <= 0 or type(lim) is not int or lim <= 0:
        raise ValueError("Invalid integer amounts")
    for k in (
        "hospitalization_covered",
        "waiting_period_satisfied",
        "documents_available",
        "invoice_matches",
        "duplicate_claim",
        "provider_flagged",
        "identity_verified",
        "consent_available",
    ):
        if type(claim_dict[k]) is not bool:
            raise ValueError(f"Expected strict bool for {k}")

    o_policy = (
        "SAT"
        if (claim_dict["hospitalization_covered"] and claim_dict["waiting_period_satisfied"] and amt <= lim)
        else "FAIL"
    )
    o_evidence = "SAT" if (claim_dict["documents_available"] and claim_dict["invoice_matches"]) else "FAIL"
    o_risk = "SAT" if (not claim_dict["duplicate_claim"] and not claim_dict["provider_flagged"]) else "FAIL"
    o_compliance = "SAT" if (claim_dict["identity_verified"] and claim_dict["consent_available"]) else "FAIL"

    obligations = {
        "policy": o_policy,
        "evidence": o_evidence,
        "risk": o_risk,
        "compliance": o_compliance,
    }

    # Precedence rules (clean inputs -> both modes coincide; computed explicitly here)
    if o_policy == "FAIL":
        label_substantive = "REJECT"
        label_strict = "REJECT"
    elif o_evidence == "FAIL" or o_risk == "FAIL" or o_compliance == "FAIL":
        label_substantive = "REVIEW"
        label_strict = "REVIEW"
    else:
        label_substantive = "APPROVE"
        label_strict = "APPROVE"

    edge_flags: list[str] = []
    if amt == lim:
        edge_flags.append("exact_coverage_limit_equality")
    elif amt == lim + 1:
        edge_flags.append("one_rupee_over_coverage_limit")
    elif amt == lim - 1:
        edge_flags.append("one_rupee_below_coverage_limit")
    if o_policy == "FAIL" and (o_evidence == "FAIL" or o_risk == "FAIL" or o_compliance == "FAIL"):
        edge_flags.append("mixed_exclusion_and_procedural_failure")

    return {
        "spec_version": SPEC_VERSION,
        "obligations": obligations,
        "substantive_exclusion_label": label_substantive,
        "strict_audit_label": label_strict,
        "label": label_substantive,
        "edge_flags": edge_flags,
    }


def _claim_fact_dict(claim: InsuranceClaim) -> dict[str, Any]:
    d = asdict(claim)
    d.pop("claim_id", None)
    return d


def _fact_signature_hash(claim: InsuranceClaim) -> str:
    return sha256(_claim_fact_dict(claim))


def _bool_signature(claim: InsuranceClaim) -> tuple[bool, ...]:
    return (
        claim.amount_inr <= claim.coverage_limit_inr,
        claim.hospitalization_covered,
        claim.waiting_period_satisfied,
        claim.documents_available,
        claim.invoice_matches,
        claim.duplicate_claim,
        claim.provider_flagged,
        claim.identity_verified,
        claim.consent_available,
    )


def _make_claim(claim_id: str, **kwargs: Any) -> InsuranceClaim:
    return InsuranceClaim(claim_id=claim_id, **kwargs)


def generate_frozen_splits(seed: int = GENERATOR_SEED) -> dict[str, Any]:
    """Deterministically generate >=120 distinct synthetic claims with strict family split isolation."""
    rng = random.Random(seed)
    dev_cases: list[dict[str, Any]] = []
    test_cases: list[dict[str, Any]] = []
    seen_fact_hashes: set[str] = set()

    def _add(split_list: list[dict[str, Any]], split: str, family_id: str, idx: int, **fields: Any) -> None:
        cid = f"p5_{split}_{family_id.lower()}_{idx:03d}"
        claim = _make_claim(cid, **fields)
        fhash = _fact_signature_hash(claim)
        if fhash in seen_fact_hashes:
            raise ValueError(f"Duplicate fact signature generated for {cid}")
        seen_fact_hashes.add(fhash)
        split_list.append({
            "claim_id": cid,
            "split": split,
            "family_id": family_id,
            "fact_hash": fhash,
            "claim": asdict(claim),
        })

    # =========================================================================
    # DEV SPLIT: 4 EXCLUSIVE TEMPLATE FAMILIES (24 CLAIMS)
    # =========================================================================
    # 1. FAM_DEV_CLEAN_MODERATE (6 APPROVE claims, utilization 50%..80%)
    for i, (amt, lim) in enumerate([
        (32_000, 60_000),
        (48_000, 80_000),
        (55_000, 100_000),
        (72_000, 120_000),
        (96_000, 160_000),
        (135_000, 200_000),
    ]):
        _add(dev_cases, "dev", "FAM_DEV_CLEAN_MODERATE", i, amount_inr=amt, coverage_limit_inr=lim)

    # 2. FAM_DEV_WAITING_EXCL (6 REJECT claims: waiting_period_satisfied=False only)
    for i, (amt, lim) in enumerate([
        (18_000, 50_000),
        (34_000, 75_000),
        (44_000, 90_000),
        (61_000, 110_000),
        (77_000, 140_000),
        (89_000, 180_000),
    ]):
        _add(
            dev_cases,
            "dev",
            "FAM_DEV_WAITING_EXCL",
            i,
            amount_inr=amt,
            coverage_limit_inr=lim,
            waiting_period_satisfied=False,
        )

    # 3. FAM_DEV_MISSING_DOCS (6 REVIEW claims: documents_available=False, invoice_matches=True)
    for i, (amt, lim) in enumerate([
        (21_000, 65_000),
        (39_000, 85_000),
        (52_000, 105_000),
        (66_000, 130_000),
        (81_000, 150_000),
        (99_000, 190_000),
    ]):
        _add(
            dev_cases,
            "dev",
            "FAM_DEV_MISSING_DOCS",
            i,
            amount_inr=amt,
            coverage_limit_inr=lim,
            documents_available=False,
        )

    # 4. FAM_DEV_MIXED_LIMIT_DUP (6 REJECT claims: over limit + duplicate_claim=True)
    for i, (amt, lim) in enumerate([
        (68_000, 50_000),
        (92_000, 75_000),
        (125_000, 100_000),
        (145_000, 120_000),
        (185_000, 150_000),
        (230_000, 200_000),
    ]):
        _add(
            dev_cases,
            "dev",
            "FAM_DEV_MIXED_LIMIT_DUP",
            i,
            amount_inr=amt,
            coverage_limit_inr=lim,
            duplicate_claim=True,
        )

    # =========================================================================
    # TEST SPLIT: 8 HELD-OUT EXCLUSIVE TEMPLATE FAMILIES (104 CLAIMS)
    # =========================================================================
    # 5. FAM_TEST_CLEAN_BOUNDARY_AND_MICRO (24 APPROVE claims: exact boundaries & low utilization)
    clean_pairs = [
        # 12 exact-equality & off-by-one boundary tiers
        (50_000, 50_000),
        (75_000, 75_000),
        (100_000, 100_000),
        (150_000, 150_000),
        (250_000, 250_000),
        (400_000, 400_000),
        (49_999, 50_000),
        (99_999, 100_000),
        (149_999, 150_000),
        (249_999, 250_000),
        (299_990, 300_000),
        (499_999, 500_000),
        # 12 low-utilization micro-tier claims (< 30%)
        (4_500, 50_000),
        (8_200, 75_000),
        (11_500, 100_000),
        (14_800, 120_000),
        (17_400, 150_000),
        (22_600, 180_000),
        (26_900, 220_000),
        (31_250, 250_000),
        (38_400, 300_000),
        (44_750, 350_000),
        (51_100, 450_000),
        (63_900, 600_000),
    ]
    for i, (amt, lim) in enumerate(clean_pairs):
        _add(test_cases, "test", "FAM_TEST_CLEAN_BOUNDARY_AND_MICRO", i, amount_inr=amt, coverage_limit_inr=lim)

    # 6. FAM_TEST_SINGLE_HOSP_EXCL (12 REJECT claims: hospitalization_covered=False only)
    for i in range(12):
        lim = 50_000 + i * 25_000
        amt = int(lim * rng.uniform(0.25, 0.90))
        _add(
            test_cases,
            "test",
            "FAM_TEST_SINGLE_HOSP_EXCL",
            i,
            amount_inr=amt,
            coverage_limit_inr=lim,
            hospitalization_covered=False,
        )

    # 7. FAM_TEST_SINGLE_OVER_LIMIT (12 REJECT claims: +1 INR boundary and severe limit excess)
    over_pairs = [
        (50_001, 50_000),
        (75_001, 75_000),
        (100_001, 100_000),
        (150_001, 150_000),
        (250_001, 250_000),
        (300_001, 300_000),
        (62_500, 50_000),
        (115_000, 80_000),
        (142_000, 100_000),
        (215_000, 150_000),
        (310_000, 200_000),
        (540_000, 350_000),
    ]
    for i, (amt, lim) in enumerate(over_pairs):
        _add(test_cases, "test", "FAM_TEST_SINGLE_OVER_LIMIT", i, amount_inr=amt, coverage_limit_inr=lim)

    # 8. FAM_TEST_MULTI_POLICY_EXCL (10 REJECT claims: multi-predicate policy failures)
    for i in range(10):
        lim = 80_000 + i * 20_000
        if i % 3 == 0:
            _add(
                test_cases,
                "test",
                "FAM_TEST_MULTI_POLICY_EXCL",
                i,
                amount_inr=lim + 12_000,
                coverage_limit_inr=lim,
                hospitalization_covered=False,
                waiting_period_satisfied=False,
            )
        elif i % 3 == 1:
            _add(
                test_cases,
                "test",
                "FAM_TEST_MULTI_POLICY_EXCL",
                i,
                amount_inr=lim + 5_000,
                coverage_limit_inr=lim,
                hospitalization_covered=False,
            )
        else:
            _add(
                test_cases,
                "test",
                "FAM_TEST_MULTI_POLICY_EXCL",
                i,
                amount_inr=lim - 10_000,
                coverage_limit_inr=lim,
                hospitalization_covered=False,
                waiting_period_satisfied=False,
            )

    # 9. FAM_TEST_REVIEW_INVOICE_EVIDENCE (12 REVIEW claims: invoice mismatch / dual evidence gap)
    for i in range(12):
        lim = 60_000 + i * 15_000
        amt = 19_000 + i * 7_500
        _add(
            test_cases,
            "test",
            "FAM_TEST_REVIEW_INVOICE_EVIDENCE",
            i,
            amount_inr=amt,
            coverage_limit_inr=lim,
            documents_available=(i < 6),
            invoice_matches=False,
        )

    # 10. FAM_TEST_REVIEW_RISK_ONLY (12 REVIEW claims: provider_flagged, duplicate_claim, or both)
    for i in range(12):
        lim = 70_000 + i * 18_000
        amt = 23_000 + i * 8_000
        dup = (i % 3) in (0, 2)
        flag = (i % 3) in (1, 2)
        _add(
            test_cases,
            "test",
            "FAM_TEST_REVIEW_RISK_ONLY",
            i,
            amount_inr=amt,
            coverage_limit_inr=lim,
            duplicate_claim=dup,
            provider_flagged=flag,
        )

    # 11. FAM_TEST_REVIEW_COMPLIANCE_MULTI (10 REVIEW claims: compliance / multi-procedural deficiencies)
    comp_patterns = [
        dict(identity_verified=False, consent_available=True, invoice_matches=True, provider_flagged=False),
        dict(identity_verified=True, consent_available=False, invoice_matches=True, provider_flagged=False),
        dict(identity_verified=False, consent_available=False, invoice_matches=True, provider_flagged=False),
        dict(identity_verified=False, consent_available=True, invoice_matches=False, provider_flagged=False),
        dict(identity_verified=True, consent_available=False, invoice_matches=False, provider_flagged=True),
        dict(identity_verified=False, consent_available=True, invoice_matches=True, provider_flagged=True),
        dict(identity_verified=True, consent_available=False, documents_available=False, invoice_matches=True),
        dict(identity_verified=False, consent_available=False, documents_available=False, invoice_matches=False),
        dict(identity_verified=False, consent_available=True, duplicate_claim=True, provider_flagged=False),
        dict(identity_verified=True, consent_available=False, duplicate_claim=True, provider_flagged=True),
    ]
    for i, pat in enumerate(comp_patterns):
        _add(
            test_cases,
            "test",
            "FAM_TEST_REVIEW_COMPLIANCE_MULTI",
            i,
            amount_inr=27_500 + i * 6_000,
            coverage_limit_inr=120_000 + i * 10_000,
            **pat,
        )

    # 12. FAM_TEST_MIXED_EXCLUSION_PROCEDURAL (12 REJECT claims: exclusion + non-policy deficiencies)
    mixed_patterns = [
        dict(hospitalization_covered=False, documents_available=False),
        dict(hospitalization_covered=False, provider_flagged=True),
        dict(hospitalization_covered=False, consent_available=False),
        dict(hospitalization_covered=False, invoice_matches=False, identity_verified=False),
        dict(waiting_period_satisfied=False, provider_flagged=True),
        dict(waiting_period_satisfied=False, invoice_matches=False),
        dict(waiting_period_satisfied=False, identity_verified=False),
        dict(waiting_period_satisfied=False, documents_available=False, consent_available=False),
        dict(amount_inr=160_000, coverage_limit_inr=100_000, consent_available=False),
        dict(amount_inr=210_000, coverage_limit_inr=150_000, invoice_matches=False),
        dict(amount_inr=280_000, coverage_limit_inr=200_000, provider_flagged=True),
        dict(amount_inr=195_000, coverage_limit_inr=125_000, documents_available=False, identity_verified=False),
    ]
    for i, pat in enumerate(mixed_patterns):
        kwargs = {
            "amount_inr": 41_000 + i * 3_500,
            "coverage_limit_inr": 100_000 + i * 10_000,
            **pat,
        }
        _add(test_cases, "test", "FAM_TEST_MIXED_EXCLUSION_PROCEDURAL", i, **kwargs)

    # Derive independent spec-v1 oracle labels & cross-check against evaluate_rule_oracle
    dev_labels: list[dict[str, Any]] = []
    test_labels: list[dict[str, Any]] = []
    for src, dst in ((dev_cases, dev_labels), (test_cases, test_labels)):
        for rec in src:
            claim_obj = InsuranceClaim(**rec["claim"])
            spec_eval = derive_spec_v1_labels(rec["claim"])
            oracle_eval = evaluate_rule_oracle(claim_obj)
            cross_ok = spec_eval["label"] == oracle_eval
            if not cross_ok:
                raise AssertionError(
                    f"Spec vs rule_oracle mismatch on {rec['claim_id']}: {spec_eval['label']} != {oracle_eval}"
                )
            dst.append({
                "claim_id": rec["claim_id"],
                "split": rec["split"],
                "family_id": rec["family_id"],
                "fact_hash": rec["fact_hash"],
                "cross_check_rule_oracle_passed": cross_ok,
                **spec_eval,
            })

    # Generate 32 update paths across held-out test cases for Phase 6 selective-recovery benchmarking
    update_paths, update_labels = _generate_update_paths(test_cases)

    return {
        "dev_inputs": dev_cases,
        "dev_labels": dev_labels,
        "test_inputs": test_cases,
        "test_labels": test_labels,
        "update_inputs": update_paths,
        "update_labels": update_labels,
    }


def _generate_update_paths(test_cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build 32 deterministic v1->v2 claim updates spanning all 4 update categories."""
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in test_cases:
        by_family[c["family_id"]].append(c)

    transitions: list[dict[str, Any]] = []
    t_labels: list[dict[str, Any]] = []

    def _rec(t_id: str, update_kind: str, base_case: dict[str, Any], delta: dict[str, Any], changed_roles: list[str]) -> None:
        v1 = dict(base_case["claim"])
        v2 = {**v1, **delta}
        l1 = derive_spec_v1_labels(v1)
        l2 = derive_spec_v1_labels(v2)
        if l1["label"] != evaluate_rule_oracle(InsuranceClaim(**v1)):
            raise AssertionError(f"Update v1 cross-check failed for {t_id}")
        if l2["label"] != evaluate_rule_oracle(InsuranceClaim(**v2)):
            raise AssertionError(f"Update v2 cross-check failed for {t_id}")
        transitions.append({
            "transition_id": t_id,
            "base_claim_id": base_case["claim_id"],
            "family_id": base_case["family_id"],
            "update_kind": update_kind,
            "expected_changed_roles": changed_roles,
            "claim_v1": v1,
            "claim_v2": v2,
        })
        t_labels.append({
            "transition_id": t_id,
            "update_kind": update_kind,
            "label_v1": l1["label"],
            "label_v2": l2["label"],
            "label_changed": l1["label"] != l2["label"],
            "obligations_v1": l1["obligations"],
            "obligations_v2": l2["obligations"],
        })

    clean_pool = by_family["FAM_TEST_CLEAN_BOUNDARY_AND_MICRO"]
    # 12 narrow single-role edits (including boundary flips & single-role repairs/degradations)
    for i in range(4):
        c = clean_pool[i]
        _rec(f"UPD_NARROW_POLICY_BOUNDARY_{i:02d}", "narrow_single_role", c, {"amount_inr": c["claim"]["coverage_limit_inr"] + 1}, ["policy"])
    for i in range(4, 8):
        c = clean_pool[i]
        _rec(f"UPD_NARROW_EVIDENCE_{i:02d}", "narrow_single_role", c, {"documents_available": False}, ["evidence"])
    for i in range(8, 10):
        c = clean_pool[i]
        _rec(f"UPD_NARROW_RISK_{i:02d}", "narrow_single_role", c, {"provider_flagged": True}, ["risk"])
    for i in range(10, 12):
        c = clean_pool[i]
        _rec(f"UPD_NARROW_COMPLIANCE_{i:02d}", "narrow_single_role", c, {"consent_available": False}, ["compliance"])

    # 8 broad multi-role edits
    for i in range(12, 16):
        c = clean_pool[i]
        _rec(
            f"UPD_BROAD_TWO_ROLE_{i:02d}",
            "broad_multi_role",
            c,
            {"documents_available": False, "provider_flagged": True},
            ["evidence", "risk"],
        )
    for i in range(4):
        c = by_family["FAM_TEST_MIXED_EXCLUSION_PROCEDURAL"][i]
        _rec(
            f"UPD_BROAD_REPAIR_ALL_{i:02d}",
            "broad_multi_role",
            c,
            {
                "hospitalization_covered": True,
                "waiting_period_satisfied": True,
                "documents_available": True,
                "invoice_matches": True,
                "duplicate_claim": False,
                "provider_flagged": False,
                "identity_verified": True,
                "consent_available": True,
            },
            sorted(r for r in ROLES if any(c["claim"][k] != True and k not in ("duplicate_claim", "provider_flagged", "amount_inr", "coverage_limit_inr") for k in []) or True),
        )
        # Recompute exact changed_roles cleanly against actual delta
        v1, v2 = transitions[-1]["claim_v1"], transitions[-1]["claim_v2"]
        c1, c2 = InsuranceClaim(**v1), InsuranceClaim(**v2)
        transitions[-1]["expected_changed_roles"] = [r for r in ROLES if c1.facts(r) != c2.facts(r)]

    # 6 no-change edits
    for i in range(16, 22):
        c = clean_pool[i]
        _rec(f"UPD_NO_CHANGE_{i:02d}", "no_change_edit", c, {}, [])

    # 6 same-label value edits (e.g. amount increases within limit on APPROVE, or deeper over-limit on REJECT)
    for i in range(22, 24):
        c = clean_pool[i]
        new_amt = c["claim"]["amount_inr"] + 5_000
        _rec(f"UPD_SAME_LABEL_APPROVE_{i:02d}", "same_label_value_edit", c, {"amount_inr": new_amt}, ["policy"])
    for i in range(4):
        c = by_family["FAM_TEST_SINGLE_OVER_LIMIT"][i]
        _rec(f"UPD_SAME_LABEL_REJECT_{i:02d}", "same_label_value_edit", c, {"amount_inr": c["claim"]["amount_inr"] + 1_000}, ["policy"])

    return transitions, t_labels


def audit_frozen_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    """Run integrity audit verifying distinct claims, family isolation, zero leakage, and class balance."""
    dev_in = dataset["dev_inputs"]
    dev_lbl = dataset["dev_labels"]
    test_in = dataset["test_inputs"]
    test_lbl = dataset["test_labels"]
    upd_in = dataset["update_inputs"]
    upd_lbl = dataset["update_labels"]

    all_in = dev_in + test_in
    all_ids = [r["claim_id"] for r in all_in]
    all_hashes = [r["fact_hash"] for r in all_in]

    dev_families = {r["family_id"] for r in dev_in}
    test_families = {r["family_id"] for r in test_in}
    overlapping_families = sorted(dev_families & test_families)

    dev_hashes = {r["fact_hash"] for r in dev_in}
    test_hashes = {r["fact_hash"] for r in test_in}
    overlapping_hashes = sorted(dev_hashes & test_hashes)

    # Check non-APPROVE boolean signature separation
    dev_non_approve_bools = {
        _bool_signature(InsuranceClaim(**r["claim"]))
        for r, l in zip(dev_in, dev_lbl)
        if l["label"] != "APPROVE"
    }
    test_non_approve_bools = {
        _bool_signature(InsuranceClaim(**r["claim"]))
        for r, l in zip(test_in, test_lbl)
        if l["label"] != "APPROVE"
    }
    overlapping_non_approve_bools = len(dev_non_approve_bools & test_non_approve_bools)

    family_rows = []
    for split_name, cases, labels in (("dev", dev_in, dev_lbl), ("test", test_in, test_lbl)):
        by_fam: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for c, l in zip(cases, labels):
            by_fam[c["family_id"]].append({"case": c, "label": l})
        for fam_id in sorted(by_fam):
            items = by_fam[fam_id]
            cls_counts = Counter(x["label"]["label"] for x in items)
            family_rows.append({
                "split": split_name,
                "family_id": fam_id,
                "count": len(items),
                "unique_fact_hashes": len({x["case"]["fact_hash"] for x in items}),
                "APPROVE": cls_counts.get("APPROVE", 0),
                "REJECT": cls_counts.get("REJECT", 0),
                "REVIEW": cls_counts.get("REVIEW", 0),
            })

    dev_class_dist = dict(Counter(l["label"] for l in dev_lbl))
    test_class_dist = dict(Counter(l["label"] for l in test_lbl))
    upd_kind_dist = dict(Counter(u["update_kind"] for u in upd_in))

    audit = {
        "spec_version": SPEC_VERSION,
        "generator_version": GENERATOR_VERSION,
        "generator_seed": GENERATOR_SEED,
        "total_claims": len(all_in),
        "unique_claim_ids": len(set(all_ids)),
        "unique_fact_hashes": len(set(all_hashes)),
        "dev_count": len(dev_in),
        "test_count": len(test_in),
        "dev_family_count": len(dev_families),
        "test_family_count": len(test_families),
        "overlapping_families": overlapping_families,
        "overlapping_family_count": len(overlapping_families),
        "overlapping_fact_hashes": overlapping_hashes,
        "overlapping_fact_hash_count": len(overlapping_hashes),
        "overlapping_non_approve_boolean_signatures": overlapping_non_approve_bools,
        "dev_class_distribution": dev_class_dist,
        "test_class_distribution": test_class_dist,
        "update_paths_count": len(upd_in),
        "update_kind_distribution": upd_kind_dist,
        "all_labels_cross_checked": all(l["cross_check_rule_oracle_passed"] for l in (dev_lbl + test_lbl)),
        "ambiguous_or_excluded_edge_cases": [
            {
                "claim_id": l["claim_id"],
                "flags": l["edge_flags"],
                "derived_label": l["label"],
            }
            for l in (dev_lbl + test_lbl)
            if l["edge_flags"]
        ],
        "family_table": family_rows,
        "integrity_passed": (
            len(all_in) >= 120
            and len(test_in) >= 100
            and len(set(all_ids)) == len(all_in)
            and len(set(all_hashes)) == len(all_in)
            and len(overlapping_families) == 0
            and len(overlapping_hashes) == 0
            and overlapping_non_approve_bools == 0
            and all(l["cross_check_rule_oracle_passed"] for l in (dev_lbl + test_lbl))
        ),
    }
    return audit


_FORBIDDEN_AUTHOR_MARKERS = frozenset({"", "synthetic", "fake", "auto", "placeholder", "bot", "gpt", "llm"})


def load_blind_external_cases(path: str | Path) -> dict[str, Any]:
    """Validate and load teammate-authored blind claims without accepting fabricated entries."""
    p = Path(path)
    if not p.exists():
        return {
            "status": "PENDING_TEAMMATE_SUBMISSION",
            "path": str(p),
            "count": 0,
            "target_count": 20,
            "cases": [],
            "sha256": None,
            "notes": "Blind external file missing; waiting for human teammate submission.",
        }
    raw_text = p.read_text(encoding="utf-8")
    data = json.loads(raw_text)
    if data.get("schema_version") != "blind-external-v1":
        raise ValueError(f"Unsupported blind external schema: {data.get('schema_version')}")

    raw_cases = data.get("cases", [])
    if not isinstance(raw_cases, list):
        raise ValueError("`cases` must be a JSON array")

    file_hash = sha256(json.loads(raw_text))
    if not raw_cases:
        return {
            "status": "PENDING_TEAMMATE_SUBMISSION",
            "path": str(p),
            "count": 0,
            "target_count": int(data.get("required_case_count_target", 20)),
            "cases": [],
            "sha256": file_hash,
            "notes": "Blind external template contains 0 submissions (`PENDING_TEAMMATE_SUBMISSION`).",
        }

    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for idx, entry in enumerate(raw_cases):
        if not isinstance(entry, dict):
            raise ValueError(f"Blind case {idx} must be a JSON object")
        for req_key in ("case_id", "author", "authored_blind_to_system_outputs", "review_status", "claim"):
            if req_key not in entry:
                raise ValueError(f"Blind case {idx} missing required field: {req_key}")
        author = str(entry["author"]).strip()
        if author.lower() in _FORBIDDEN_AUTHOR_MARKERS:
            raise ValueError(f"Blind case {idx} has invalid/non-human author marker: '{author}'")
        if entry["authored_blind_to_system_outputs"] is not True:
            raise ValueError(f"Blind case {idx} must assert authored_blind_to_system_outputs == true")
        cid = str(entry["case_id"]).strip()
        if not cid or cid in seen_ids:
            raise ValueError(f"Duplicate or empty blind case_id: '{cid}'")
        seen_ids.add(cid)
        claim = InsuranceClaim(**entry["claim"])
        label_info = derive_spec_v1_labels(asdict(claim))
        validated.append({
            "case_id": cid,
            "author": author,
            "authored_blind_to_system_outputs": True,
            "review_status": str(entry["review_status"]),
            "claim": asdict(claim),
            "spec_v1_label": label_info["label"],
        })

    return {
        "status": "READY" if len(validated) >= int(data.get("required_case_count_target", 20)) else "PARTIAL_SUBMISSION",
        "path": str(p),
        "count": len(validated),
        "target_count": int(data.get("required_case_count_target", 20)),
        "cases": validated,
        "sha256": file_hash,
        "notes": "Human-authored blind cases validated.",
    }
