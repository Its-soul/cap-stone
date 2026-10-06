"""Explicit offline synthetic milestone demo. Does not use V2/V3 execution paths."""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cert_recovery.adjudication.domain import InsuranceClaim
from cert_recovery.adjudication.experiments import comparison, run_experiments
from cert_recovery.adjudication.ledger import EventDAG
from cert_recovery.adjudication.workflow import InsuranceAdjudication


def demonstrate():
    healthy = InsuranceAdjudication(InsuranceClaim()).export()
    faulty = InsuranceAdjudication(InsuranceClaim(), faults={"risk_agent": "failure"})
    before = faulty.export()
    outcome = faulty.recover_agent("risk_agent")
    records = copy.deepcopy(healthy["events"])
    valid_before = EventDAG.verify(records, healthy["dag_checkpoint"])
    records[0]["payload"]["tampered"] = True
    return {"synthetic": True, "insurance": healthy, "adversarial": before,
            "tamper_detection": {"before": valid_before, "after": EventDAG.verify(records, healthy["dag_checkpoint"])},
            "recovery": {"status": outcome.status, "executions": faulty.executions, "final": faulty.result()},
            "comparison": comparison()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", action="store_true")
    parser.add_argument("--trials", type=int, default=20, help="Trials per requested rate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="New JSON file; existing files are never overwritten")
    args = parser.parse_args()
    result = run_experiments(trials=args.trials, seed=args.seed) if args.experiments else demonstrate()
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
    summary = result["summaries"] if args.experiments else {
        "insurance_decision": result["insurance"]["final"]["decision"],
        "tamper_detection": result["tamper_detection"], "recovery": result["recovery"],
        "comparison": {k: {"baseline": v["baseline"], "proposed": v["proposed"]} for k, v in result["comparison"].items()}}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
