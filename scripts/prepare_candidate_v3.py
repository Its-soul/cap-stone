"""Prepare annotation sidecars and disabled candidate-v3 proposal, offline only."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cert_recovery.annotation_review import prepare_bundle, write_exclusive
from cert_recovery.pinned_smoke import sha256_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    out=args.output_dir.resolve()
    if out.exists() and any(p.name!='historical_hashes.json' for p in out.iterdir()):
        raise FileExistsError('Use a new review directory; preserve completed or failed attempts')
    out.mkdir(parents=True,exist_ok=True)
    def blocked(*a,**kw): raise AssertionError('Preparation prohibits all network connections')
    socket.socket.connect=socket.socket.connect_ex=blocked
    if not (out/'historical_hashes.json').exists():
        files=[p for p in (ROOT/'_private/runs').rglob('*') if p.is_file()]
        files += [ROOT/'baseline_tests.json',*list((ROOT/'notebooks').glob('*.ipynb')),
                  *list((ROOT/'_private/research/notebooks').glob('*.ipynb')),
                  *list((ROOT/'_private/research/results').glob('*.json'))]
        files += list((ROOT/'src').rglob('*.py'))
        write_exclusive(out/'historical_hashes.json',{str(p.relative_to(ROOT)):sha256_file(p) for p in files})
    try:
        summary=prepare_bundle(ROOT,out)
    except Exception as error:
        write_exclusive(out/'preparation_failure.json',{'status':'ERROR','type':type(error).__name__,
            'message':str(error),'utc':datetime.now(timezone.utc).isoformat(),'no_model_execution':True})
        raise
    sources=['src/cert_recovery/dependency_contract.py','src/cert_recovery/contrast_data.py',
             'src/cert_recovery/annotation_review.py','scripts/prepare_candidate_v3.py',
             'scripts/validate_annotation_review.py','tests/test_dependency_contract.py',
             'tests/test_contrast_data.py','tests/test_annotation_review.py']
    write_exclusive(out/'phase_manifest.json',{'status':'FROZEN_PREPARATION_ONLY_NO_MODEL_EXECUTION',
        'prepared_artifact_hashes':{str(p.relative_to(out)):sha256_file(p) for p in out.rglob('*') if p.is_file()},
        'preparation_source_hashes':{p:sha256_file(ROOT/p) for p in sources}})
    print(json.dumps({'status':'PREPARED_NOT_EXECUTED','summary':summary,'report':str(out/'report.html')},indent=2))


if __name__=='__main__': main()
