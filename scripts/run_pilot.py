"""Explicit bounded local/Colab synthetic pilot. No work without --execute."""
import argparse
from pathlib import Path
import subprocess
import sys
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cert_recovery.pilot import create_run, execute_stage, finalize
from cert_recovery.data_pipeline import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Authorizes one bounded pilot, weights and training")
    parser.add_argument("--stage", choices=["data","system","deberta","minilm_fallback","pretrained","training","evaluation"])
    parser.add_argument("--run", type=Path)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        if args.run is None:
            parser.error("--report-only needs --run")
        finalize(args.run)
        return
    if not args.execute:
        parser.error("No execution authorized: pass --execute deliberately")
    if args.stage:
        if args.run is None:
            parser.error("--stage requires --run")
        state = execute_stage(args.run, args.stage)
        print(args.stage, state["status"], flush=True)
        raise SystemExit(0 if state["status"] == "SUCCESS" else 1)
    if args.run:
        parser.error("A complete pilot always creates a new directory; stage replay requires --stage")
    run = create_run(ROOT)
    print(str(run), flush=True)
    env = dict(os.environ, HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_XET="1",
               HF_HUB_DOWNLOAD_TIMEOUT="120", HF_HUB_ETAG_TIMEOUT="30", TOKENIZERS_PARALLELISM="false",
               HF_HOME=str(ROOT / "_private/runtime/hf-cache"), MPLBACKEND="Agg")
    for stage in ("data","system","deberta","minilm_fallback","pretrained","training","evaluation"):
        import json
        prerequisite = "data" if stage in ("system","deberta","minilm_fallback","pretrained","training") else "training" if stage == "evaluation" else None
        if prerequisite and json.loads((run / (prerequisite+".status.json")).read_text())["status"] != "SUCCESS":
            write_json(run / (stage+".status.json"), {"status":"BLOCKED","reason":prerequisite+" failed; no downstream execution"})
            continue
        if stage in ("pretrained","training") and json.loads((run / "deberta.status.json").read_text())["status"] != "SUCCESS":
            write_json(run / (stage+".status.json"), {"status":"BLOCKED","reason":"Pinned DeBERTa smoke failed; no repeated model loading attempts"})
            continue
        if stage == "evaluation":
            if json.loads((run / "training.status.json").read_text())["status"] != "SUCCESS":
                continue
        with (run / (stage + ".console.log")).open("w", encoding="utf-8") as stream:
            try:
                outcome = subprocess.run([sys.executable,str(Path(__file__).resolve()),"--execute","--run",str(run),"--stage",stage],
                                         env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
                print(stage, "SUCCESS" if outcome.returncode == 0 else "ERROR", flush=True)
            except subprocess.TimeoutExpired:
                write_json(run / (stage+".status.json"), {"status":"TIMEOUT","error":{"type":"TimeoutExpired","message":"Bounded 1800 second stage limit exceeded"}})
                print(stage, "TIMEOUT", flush=True)
        finalize(run)
    finalize(run)
    print("Report:", run / "report.html", flush=True)


if __name__ == "__main__":
    main()
