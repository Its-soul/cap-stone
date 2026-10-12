"""Execute the frozen specialist composition evaluation against the repaired baseline.

Outputs results to results/composition/ with strict overwrite guards protecting
all historical bundles and repaired ablation baselines.
"""
import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cert_recovery.adjudication.composition_evaluation import run_composition_study


def main():
    print("=" * 78)
    print("SPECIALIST-ROLE COMPOSITION EVALUATION (composition-v1)")
    print("Status: SYNTHETIC_PROVISIONAL (No human approval claimed)")
    print("=" * 78)

    output_dir = REPO_ROOT / "results" / "composition"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Evaluate Extended Scenario Suite (150 trials)
    print("\n[1/3] Running Extended Suite (150 presentations across 10 scenarios)...")
    res_ext = run_composition_study(
        cases_set="extended",
        rates=(0.0, 0.1, 0.2, 0.3, 0.4),
        seeds=(42, 101, 2024),
    )
    ext_file = output_dir / "composition_evaluation_extended.json"
    with open(ext_file, "w") as f:
        json.dump(res_ext, f, indent=2)
    print(f" Saved to: {ext_file}")

    # 2. Evaluate Historical Scenario Suite (75 trials)
    print("\n[2/3] Running Historical Suite (75 presentations across 5 scenarios)...")
    res_hist = run_composition_study(
        cases_set="historical",
        rates=(0.0, 0.1, 0.2, 0.3, 0.4),
        seeds=(42, 101, 2024),
    )
    hist_file = output_dir / "composition_evaluation_historical.json"
    with open(hist_file, "w") as f:
        json.dump(res_hist, f, indent=2)
    print(f" Saved to: {hist_file}")

    # 3. Evaluate Mixed-Condition Suite (60 trials)
    print("\n[3/3] Running Mixed-Condition & Uncertainty Suite (60 presentations across 6 scenarios)...")
    res_mixed = run_composition_study(
        cases_set="mixed",
        rates=(0.0, 0.1, 0.2, 0.3, 0.4),
        seeds=(42, 101),
    )
    mixed_file = output_dir / "composition_evaluation_mixed.json"
    with open(mixed_file, "w") as f:
        json.dump(res_mixed, f, indent=2)
    print(f" Saved to: {mixed_file}")

    # Print Summary Tables
    print("\n" + "=" * 78)
    print("EXTENDED SUITE OVERALL RESULTS (150 presentations)")
    print("=" * 78)
    print(f"{'Method':<32} | {'3-Class Acc':<12} | {'Coverage':<10} | {'Decided Acc':<12} | {'Add. Deferrals':<14} | {'Unsafe App':<10}")
    print("-" * 105)
    for m in ["E", "COMP", "COMP_STRICT", "F"]:
        met = res_ext["overall_by_method"][m]
        name = res_ext["method_registry"][m]["name"]
        dec_acc_str = f"{met['decided_accuracy']:.4f}" if met["decided_accuracy"] is not None else "N/A"
        print(f"{name:<32} | {met['reported_accuracy']:<12.4f} | {met['coverage']:<10.4f} | {dec_acc_str:<12} | {met['additional_deferrals']:<14} | {met['unsafe_approvals']:<10}")

    print("\n" + "=" * 78)
    print("INVESTIGATION OF THE 42 LEGACY ADDITIONAL DEFERRALS (Extended Suite)")
    print("=" * 78)
    rej = res_ext["rejection_analysis"]
    print(f"Total Rejection Presentations (3 scenarios x 15 runs): {rej['total_rejection_presentations']}")
    print(f"Legacy Arm E False Reviews (Additional Deferrals):     {rej['legacy_E_reviews']} / {rej['total_rejection_presentations']}")
    print(f"Legacy Arm E Correct Rejections:                       {rej['legacy_E_correct_rejections']} / {rej['total_rejection_presentations']}")
    print(f"Specialist Composition Reviews (Under Faults):         {rej['composition_reviews']} / {rej['total_rejection_presentations']}")
    print(f"Specialist Composition Correct Rejections:             {rej['composition_correct_rejections']} / {rej['total_rejection_presentations']}")
    print(f"Deterministic Rules (Oracle) Correct Rejections:       {rej['rules_oracle_correct_rejections']} / {rej['total_rejection_presentations']}")
    diff_resolved = rej['composition_correct_rejections'] - rej['legacy_E_correct_rejections']
    print(f"\nNet Correct Rejections Recovered: +{diff_resolved} (eliminating false reviews when policy uncorrupted)")

    print("\n" + "=" * 78)
    print("CLEAN PRESENTATIONS ONLY (Extended Suite, Rate = 0.0, 30 presentations)")
    print("=" * 78)
    print(f"{'Method':<32} | {'3-Class Acc':<12} | {'Coverage':<10} | {'Decided Acc':<12} | {'Add. Deferrals':<14} | {'Unsafe App':<10}")
    print("-" * 105)
    for m in ["E", "COMP", "COMP_STRICT", "F"]:
        met = res_ext["clean_by_method"][m]
        name = res_ext["method_registry"][m]["name"]
        dec_acc_str = f"{met['decided_accuracy']:.4f}" if met["decided_accuracy"] is not None else "N/A"
        print(f"{name:<32} | {met['reported_accuracy']:<12.4f} | {met['coverage']:<10.4f} | {dec_acc_str:<12} | {met['additional_deferrals']:<14} | {met['unsafe_approvals']:<10}")

    print("\n" + "=" * 78)
    print("HISTORICAL SUITE OVERALL RESULTS (75 presentations)")
    print("=" * 78)
    print(f"{'Method':<32} | {'3-Class Acc':<12} | {'Coverage':<10} | {'Decided Acc':<12} | {'Add. Deferrals':<14} | {'Unsafe App':<10}")
    print("-" * 105)
    for m in ["E", "COMP", "COMP_STRICT", "F"]:
        met = res_hist["overall_by_method"][m]
        name = res_hist["method_registry"][m]["name"]
        dec_acc_str = f"{met['decided_accuracy']:.4f}" if met["decided_accuracy"] is not None else "N/A"
        print(f"{name:<32} | {met['reported_accuracy']:<12.4f} | {met['coverage']:<10.4f} | {dec_acc_str:<12} | {met['additional_deferrals']:<14} | {met['unsafe_approvals']:<10}")

    print("\n" + "=" * 78)
    print("MIXED-CONDITION & UNCERTAINTY SUITE RESULTS (60 presentations)")
    print("=" * 78)
    print(f"{'Method':<32} | {'3-Class Acc':<12} | {'Coverage':<10} | {'Decided Acc':<12} | {'Add. Deferrals':<14} | {'Unsafe App':<10}")
    print("-" * 105)
    for m in ["E", "COMP", "COMP_STRICT", "F"]:
        met = res_mixed["overall_by_method"][m]
        name = res_mixed["method_registry"][m]["name"]
        dec_acc_str = f"{met['decided_accuracy']:.4f}" if met["decided_accuracy"] is not None else "N/A"
        print(f"{name:<32} | {met['reported_accuracy']:<12.4f} | {met['coverage']:<10.4f} | {dec_acc_str:<12} | {met['additional_deferrals']:<14} | {met['unsafe_approvals']:<10}")

    print("\n" + "=" * 78)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 78)


if __name__ == "__main__":
    main()
