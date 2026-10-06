"""Prepare cache-only v3 audits/review/launch bindings, never train or infer."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def hardware():
    result = {'platform': platform.platform(), 'python': platform.python_version(),
              'logical_processors': os.cpu_count(), 'planned_device': 'CPU', 'gpu_execution_verified': False}
    if platform.system() == 'Windows':
        command = "$p=Get-CimInstance Win32_Processor; $m=Get-CimInstance Win32_ComputerSystem; [PSCustomObject]@{cpu=$p.Name;physical_cores=$p.NumberOfCores;memory_bytes=$m.TotalPhysicalMemory}|ConvertTo-Json -Compress"
        done = subprocess.run(['powershell', '-NoProfile', '-Command', command], capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)
        if done.returncode == 0: result.update(json.loads(done.stdout))
        else: result['hardware_query_status'] = 'UNAVAILABLE; no fabricated capacity'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve(); review = args.review_dir.resolve()
    if out.exists() and any(p.name != 'historical_hashes.json' for p in out.iterdir()):
        raise FileExistsError('Use a new directory; prior preparation artifacts are immutable')
    out.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HOME=str(ROOT/'_private/runtime/hf-cache'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false')
    def blocked(*a, **kw): raise AssertionError('Prerequisite preparation prohibits network')
    socket.socket.connect = socket.socket.connect_ex = blocked
    from cert_recovery.annotation_review import write_exclusive
    from cert_recovery.pinned_smoke import sha256_file
    from cert_recovery.v3_preparation import prepare_bundle
    if not (out/'historical_hashes.json').exists():
        history = json.loads((review/'historical_hashes.json').read_text())
        history.update({str(p.relative_to(ROOT)): sha256_file(p) for p in review.rglob('*') if p.is_file()})
        write_exclusive(out/'historical_hashes.json', history)
    try:
        launch = prepare_bundle(ROOT, review, out, hardware())
        sources = ['src/cert_recovery/v3_preparation.py', 'src/cert_recovery/v3_launch.py',
            'scripts/prepare_v3_prerequisites.py', 'scripts/run_candidate_v3.py', 'scripts/validate_v3_prerequisites.py',
            'tests/test_v3_preparation.py', 'tests/test_v3_launch.py',
            'src/cert_recovery/model_pipeline.py', 'src/cert_recovery/data_pipeline.py']
        source_digests = {}
        for filename in sources:
            target = out/'source_snapshot'/filename; target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream: stream.write((ROOT/filename).read_bytes())
            source_digests[filename] = sha256_file(target)
        write_exclusive(out/'preparation_manifest.json', {'status': 'PREPARED_DISABLED_PENDING_HUMAN',
            'artifact_sha256': {str(p.relative_to(out)): sha256_file(p) for p in out.rglob('*') if p.is_file()},
            'source_sha256': source_digests, 'training': False, 'model_inference': False, 'downloads': False,
            'notebook_execution': False, 'space_requests': False, 'human_reviewed': False})
    except Exception as error:
        write_exclusive(out/'preparation_failure.json', {'status': 'ERROR', 'type': type(error).__name__,
            'message': str(error), 'utc': datetime.now(timezone.utc).isoformat(), 'model_execution': False})
        raise
    print(json.dumps({'status': launch['status'], 'tokenizer_gate': launch['tokenizer_gate'],
                      'technical_blockers': launch['technical_blockers'], 'bundle': str(out)}, indent=2))


if __name__ == '__main__': main()
