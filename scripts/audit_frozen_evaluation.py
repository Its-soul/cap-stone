"""Audit an existing frozen bundle offline, without rerunning inference."""
import argparse
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cert_recovery.evidence_audit import run_audit
from cert_recovery.pinned_smoke import sha256_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a new empty audit directory; preserve prior attempts')
    out.mkdir(parents=True, exist_ok=True)
    files = [p for p in (ROOT / '_private/runs').rglob('*') if p.is_file()
             and p.suffix in {'.json', '.jsonl', '.md', '.html'} and not p.is_relative_to(out)]
    files += [ROOT / 'baseline_tests.json', *list((ROOT / '_private/research').rglob('*.json')),
              *list((ROOT / '_private/research/notebooks').glob('*.ipynb')),
              *list((ROOT / 'notebooks').glob('*.ipynb')),
              *list((ROOT / '_private/runs/pilot_20261005T095053_580194Z_02f39b02/checkpoints/best').iterdir())]
    (out / 'historical_hashes.json').write_text(json.dumps(
        {str(p.relative_to(ROOT)): sha256_file(p) for p in files}, indent=2) + '\n', encoding='utf-8')
    def blocked(*a, **kw):
        raise AssertionError('Audit prohibits network connections')
    socket.socket.connect = socket.socket.connect_ex = blocked
    run_audit(ROOT, out)
