"""Future bounded v3 job; external human review AND authorization are mandatory."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--attempt-dir', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--approval-file', type=Path)
    parser.add_argument('--authorization-file', type=Path)
    parser.add_argument('--internal-stage', choices=['training', 'comparison'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.execute: parser.error('Disabled by default; --execute plus external review/authorization required')
    if args.approval_file is None or args.authorization_file is None:
        parser.error('External approval and training authorization files are required; templates are not approval')
    os.environ.update(HF_HOME=str(ROOT/'_private/runtime/hf-cache'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false')
    def blocked(*a, **kw): raise AssertionError('Future local v3 job prohibits network/downloads/Space calls')
    socket.socket.connect = socket.socket.connect_ex = blocked
    from cert_recovery.annotation_review import write_exclusive
    from cert_recovery.pinned_smoke import sha256_file
    from cert_recovery.v3_launch import initialize_attempt, run_worker
    bundle = args.bundle.resolve(); attempt = args.attempt_dir.resolve()
    if args.internal_stage:
        print(json.dumps(run_worker(ROOT, bundle, attempt, args.internal_stage,
                                    args.approval_file, args.authorization_file)), flush=True)
        return 0
    spec = initialize_attempt(ROOT, bundle, attempt, args.approval_file, args.authorization_file)
    for stage in ['training', 'comparison']:
        folder = attempt/stage; folder.mkdir(exist_ok=False)
        write_exclusive(folder/'started.json', {'status': 'RUNNING', 'started_utc': datetime.now(timezone.utc).isoformat()})
        command = [sys.executable, str(Path(__file__).resolve()), '--bundle', str(bundle), '--attempt-dir', str(attempt),
                   '--execute', '--approval-file', str(args.approval_file.resolve()),
                   '--authorization-file', str(args.authorization_file.resolve()), '--internal-stage', stage]
        start = perf_counter(); status = 'ERROR'; error = None
        with (folder/'console.log').open('x', encoding='utf-8') as stream:
            try:
                completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                    timeout=spec['bounded_runtime'][stage+'_timeout_seconds'])
                status = 'SUCCESS' if completed.returncode == 0 else 'ERROR'
                if completed.returncode: error = {'type': 'WorkerError', 'exit_code': completed.returncode}
            except subprocess.TimeoutExpired:
                status = 'TIMEOUT'; error = {'type': 'TimeoutExpired', 'budget_seconds': 1800}
        finished = {'status': status, 'elapsed_seconds': perf_counter()-start,
                    'finished_utc': datetime.now(timezone.utc).isoformat(), 'error': error}
        if stage == 'training' and status == 'SUCCESS':
            finished['selection_frozen_sha256'] = sha256_file(attempt/'selection_frozen.json')
        write_exclusive(folder/'finished.json', finished)
        print(stage, status, flush=True)
        if status != 'SUCCESS':
            print('Stopped; failed artifacts preserved, no retry or downstream stage', flush=True)
            return 1
    return 0


if __name__ == '__main__': raise SystemExit(main())
