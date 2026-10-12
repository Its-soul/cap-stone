"""Command-line runner for the 6-arm offline insurance ablation study."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cert_recovery.adjudication.ablation import run_ablation_study, save_ablation_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["extended", "historical"], default="extended",
                        help="Case suite to evaluate: extended (10 scenarios) or historical (5 cases)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 2024],
                        help="List of random seeds for Bernoulli corruption sampling")
    parser.add_argument("--rates", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4],
                        help="Requested corruption rates to test")
    parser.add_argument("--output", type=Path, default=Path("results/ablation/ablation_study_results.json"),
                        help="Target output path (must not exist)")
    args = parser.parse_args()

    print(f"Executing offline ablation study: suite={args.suite}, seeds={args.seeds}, rates={args.rates}")
    results = run_ablation_study(cases_set=args.suite, rates=tuple(args.rates), seeds=tuple(args.seeds))

    if args.output:
        saved = save_ablation_results(results, args.output)
        print(f"Saved complete ablation records to {saved}")

    print("\n=== OVERALL COMPARISON (Arms A - I + Reference F) ===")
    print(f"{'Arm':<4} {'Method':<36} {'Accuracy (All)':<16} {'Coverage':<12} {'Decided Acc':<14} {'Unsafe Appr':<12} {'False Rev':<12}")
    print("-" * 110)
    for m in ["A", "B", "C", "D", "E", "F", "G", "H", "I"]:
        meta = results["method_registry"][m]
        stats = results["overall_by_method"][m]
        acc_str = f"{stats['reported_accuracy']:.1%} ({stats['reported_correct']}/{stats['total_cases']})"
        cov_str = f"{stats['coverage']:.1%}"
        d_acc = f"{stats['decided_accuracy']:.1%}" if stats['decided_accuracy'] is not None else "N/A"
        unsafe_str = f"{stats['unsafe_approvals']}"
        false_rev_str = f"{stats['false_reviews']}"
        print(f"{m:<4} {meta['name']:<36} {acc_str:<16} {cov_str:<12} {d_acc:<14} {unsafe_str:<12} {false_rev_str:<12}")

    print("\n=== SINGLE-FACTOR MATCHED COMPARISONS ===")
    for factor_name, factor_data in results.get("single_factor_effects", {}).items():
        print(f"\n--- {factor_data['description']} ---")
        for pair_name, pstats in factor_data["pairs"].items():
            delta_acc = pstats["delta_accuracy"] * 100
            print(f"  {pair_name:<24}: delta accuracy = {delta_acc:+.1f}%, delta unsafe = {pstats['delta_unsafe']:+d}")

    print("\n=== PAIRED DISAGREEMENTS ===")
    for pair, data in results["paired_comparisons"].items():
        print(f"  {pair}: {data['disagreement_count']}/{data['total_comparisons']} ({data['disagreement_rate']:.1%}) | transitions: {data['transitions']}")


if __name__ == "__main__":
    main()
