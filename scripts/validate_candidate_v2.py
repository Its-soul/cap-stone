"""Offline validation of a completed controlled-candidate bundle; no models loaded."""
import argparse
import json
import math
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cert_recovery.controlled_study import verify_frozen, STRESS
from cert_recovery.controlled_data import leakage_audit
from cert_recovery.crypto import sha256
from cert_recovery.data_pipeline import read_jsonl
from cert_recovery.evidence_audit import fit_baselines, baseline_predictions
from cert_recovery.paired_metrics import report_predictions
from cert_recovery.pinned_smoke import sha256_file
from cert_recovery.model_pipeline import select_threshold


def validate(run):
    run = Path(run).resolve()
    def blocked(*a, **kw):
        raise AssertionError('Bundle validation prohibits network connections')
    socket.socket.connect = socket.socket.connect_ex = blocked
    config, protocol = verify_frozen(run)
    assert sha256_file(ROOT / STRESS) == protocol['stress_file_sha256']
    rows = read_jsonl(run / 'dependency_pairs.jsonl')
    pairs = json.loads((run / 'paired_links.json').read_text())['pairs']
    stress = read_jsonl(ROOT / STRESS)
    assert leakage_audit(rows, pairs, stress) == json.loads((run / 'leakage_audit.json').read_text())
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] == 'SUCCESS'
    assert set(manifest['stages']) == {'training', 'proxy_selection', 'comparison'}
    for stage, state in manifest['stages'].items():
        assert state['status'] == 'SUCCESS'
        assert json.loads((Path(state['attempt']) / 'status.json').read_text()) == state
        assert 0 < state['elapsed_seconds'] < 1800
    selection = json.loads((run / 'checkpoints/selection.json').read_text())
    for filename, digest in selection['checkpoint_file_hashes'].items():
        assert sha256_file(run / 'checkpoints/best' / filename) == digest
    assert selection['actual_optimizer_steps'] == 256
    assert selection['config_hash'] == protocol['config_sha256']
    assert selection['split_manifest_hash'] == protocol['split_manifest_sha256']
    assert len([h for h in selection['log_history'] if 'loss' in h]) == 256
    assert [h['step'] for h in selection['log_history'] if 'eval_macro_f1' in h] == [128, 256]
    validation = selection['validation']['predictions']
    assert selection['threshold'] == select_threshold([p['label'] for p in validation],
        [p['probability'] for p in validation], protocol['model']['threshold_candidates'])
    assert selection['loaded_provenance']['head'].startswith('new two-class')
    assert len(selection['loaded_provenance']['fresh_head_initial_sha256']) == 64
    assert selection['requested_revision'] == protocol['model']['revision']
    original = json.loads((ROOT / '_private/runs/pilot_20261005T095053_580194Z_02f39b02/checkpoints/selection.json').read_text())
    assert selection['loaded_provenance']['base_weight_sha256'] == original['loaded_provenance']['base_weight_sha256']
    proxy = json.loads((run / 'metrics/pretrained_selection.json').read_text())
    frozen = json.loads((run / 'comparison_selection_frozen.json').read_text())
    assert frozen['candidate_selection_sha256'] == sha256(selection)
    assert frozen['original_selection_sha256'] == sha256(original)
    assert frozen['proxy_selection_sha256'] == sha256(proxy)
    fitted = fit_baselines([r for r in rows if r['split'] == 'train'])
    assert fitted == json.loads((run / 'baseline_fit.json').read_text())
    stress_pairs = json.loads((Path(protocol['audit_dir']) / 'paired_links.json').read_text())['pairs']
    datasets = {'validation': [r for r in rows if r['split'] == 'validation'], 'stress': stress,
                'final': [r for r in rows if r['split'] == 'test']}
    thresholds = {'original': original['threshold'], 'candidate_v2': selection['threshold'], 'nli_proxy': proxy['threshold'], 'constant': .5, 'identifier': .5}
    captured = {}
    for split, items in datasets.items():
        ids = {r['pair_id'] for r in items}
        links = stress_pairs if split == 'stress' else [p for p in pairs if set(p['members']) <= ids]
        captured[split] = {}
        for method, threshold in thresholds.items():
            record = json.loads((run / 'comparisons' / f'{split}_{method}.json').read_text())
            computed = report_predictions(items, record['predictions'], links, threshold)
            for key in ('classification', 'macro_f1', 'confusion_matrix', 'class_counts', 'paired_metrics', 'per_scenario'):
                assert record[key] == computed[key], (split, method, key)
            if method in ('constant', 'identifier'):
                predictions, coverage = baseline_predictions(items, fitted, method == 'identifier')
                assert record['baseline_provenance']['coverage'] == coverage
                assert [p['probability'] for p in predictions] == [p['probability'] for p in record['predictions']]
            else:
                for p in record['predictions']:
                    logits, probs = p['raw_logits'], p['original_model_class_probabilities']
                    assert len(logits) == (3 if method == 'nli_proxy' else 2)
                    assert all(math.isfinite(x) for x in logits)
                    exp = [math.exp(x - max(logits)) for x in logits]
                    expected = [x / sum(exp) for x in exp]
                    assert all(math.isclose(a, b, abs_tol=1e-6) for a,b in zip(expected, probs.values()))
                    positive = 'entailment' if method == 'nli_proxy' else 'depends_on'
                    assert math.isclose(probs[positive], p['probability'], abs_tol=1e-7)
                assert record['provenance']['tokenization']['truncated_rows'] == 0
            captured[split][method] = record
    differences = json.loads((run / 'differences.json').read_text())
    for split, records in captured.items():
        orig = {p['pair_id']: p for p in records['original']['predictions']}
        cand = {p['pair_id']: p for p in records['candidate_v2']['predictions']}
        assert {p['pair_id'] for p in differences[split]['improvements']} == {k for k in orig if not orig[k]['correct'] and cand[k]['correct']}
        assert {p['pair_id'] for p in differences[split]['regressions']} == {k for k in orig if orig[k]['correct'] and not cand[k]['correct']}
    selected_val = {p['pair_id']: p for p in validation}
    for p in captured['validation']['candidate_v2']['predictions']:
        assert math.isclose(p['probability'], selected_val[p['pair_id']]['probability'], abs_tol=1e-6)
    page = (run / 'report.html').read_text()
    assert 'PRELIMINARY / EXPERIMENTAL' in page and 'regressions' in page
    for path in run.glob('*.png'):
        assert path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    print('PASS: frozen 256/64/96 data, source/config/weight/tokenizer hashes, 256 optimizer steps, 15 comparison artifacts (2,720 predictions), logits/probabilities, thresholds, coverage and regression/improvement IDs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    validate(parser.parse_args().run)
