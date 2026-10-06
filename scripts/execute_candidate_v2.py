"""Run the three pre-frozen stages once, in bounded isolated subprocesses."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cert_recovery.data_pipeline import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    if not args.execute:
        parser.error('Disabled by default; --execute required')
    run = args.run.resolve()
    for stage in ('training', 'proxy_selection', 'comparison'):
        log = run / (stage + '.console.log')
        if log.exists():
            raise FileExistsError('Existing stage log must be preserved')
        start = perf_counter()
        with log.open('x', encoding='utf-8') as stream:
            try:
                result = subprocess.run([sys.executable, str(ROOT / 'scripts/run_candidate_v2.py'),
                    '--execute', '--run', str(run), '--stage', stage], stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
            except subprocess.TimeoutExpired:
                manifest = json.loads((run / 'manifest.json').read_text())
                state = manifest['stages'][stage]
                state.update(status='TIMEOUT', elapsed_seconds=perf_counter()-start,
                    finished_utc=datetime.now(timezone.utc).isoformat(),
                    error={'type': 'TimeoutExpired', 'message': 'Bounded 1800-second stage budget exhausted'})
                write_json(Path(state['attempt']) / 'status.json', state)
                manifest['status'] = 'TIMEOUT'
                write_json(run / 'manifest.json', manifest)
                print(stage, 'TIMEOUT; downstream stages not run', flush=True)
                return 1
        print(stage, 'SUCCESS' if result.returncode == 0 else 'ERROR', flush=True)
        if result.returncode:
            print('Stopped; failed artifacts preserved; no retry', flush=True)
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
