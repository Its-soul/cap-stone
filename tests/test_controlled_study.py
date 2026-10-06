import json
from pathlib import Path
import subprocess
import sys
import pytest
from cert_recovery.controlled_study import freeze


def test_protocol_requires_successful_checks(tmp_path):
    audit = tmp_path / 'audit'
    audit.mkdir()
    (audit / 'audit.json').write_text(json.dumps({'status': 'VERIFIED_OFFLINE'}))
    with pytest.raises(ValueError, match='all offline checks'):
        freeze(tmp_path, audit, {'status': 'FAILED'})
    assert not (tmp_path / '_private').exists()


@pytest.mark.parametrize('script', ['run_candidate_v2.py', 'execute_candidate_v2.py'])
def test_execution_disabled_by_default(tmp_path, script):
    path = Path(__file__).resolve().parents[1] / 'scripts' / script
    args = [sys.executable, str(path), '--run', str(tmp_path)]
    if script == 'run_candidate_v2.py':
        args += ['--stage', 'training']
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode != 0 and 'Disabled by default' in result.stderr
    assert list(tmp_path.iterdir()) == []
