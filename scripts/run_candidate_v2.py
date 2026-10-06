"""Explicit stages for a preflight-gated candidate experiment; cache-only CPU."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--stage', choices=['training', 'proxy_selection', 'comparison'], required=True)
    args = parser.parse_args()
    if not args.execute:
        parser.error('Disabled by default; explicit --execute authorization required')
    os.environ.update(HF_HOME=str(ROOT / '_private/runtime/hf-cache'), HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1', TOKENIZERS_PARALLELISM='false', MPLBACKEND='Agg')
    # Offline cached model execution has no remote inference or downloads.
    import socket
    def blocked(*a, **kw):
        raise AssertionError('Candidate study prohibits network access')
    socket.socket.connect = socket.socket.connect_ex = blocked
    from cert_recovery.controlled_study import execute_stage
    print(json.dumps(execute_stage(args.run, args.stage)), flush=True)


if __name__ == '__main__':
    main()
