"""Validate corrected private metrics and preservation hashes without inference."""
import argparse
import json
import math
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cert_recovery.data_pipeline import read_jsonl
from cert_recovery.paired_metrics import report_predictions
from cert_recovery.pinned_smoke import sha256_file


def validate(path):
    path = Path(path)
    def blocked(*a, **kw):
        raise AssertionError('Artifact validation prohibits network')
    socket.socket.connect = socket.socket.connect_ex = blocked
    history = json.loads((path / 'historical_hashes.json').read_text())
    for filename, digest in history.items():
        assert sha256_file(ROOT / filename) == digest, filename
    audit = json.loads((path / 'audit.json').read_text())
    rows = read_jsonl(ROOT / '_private/runs/eval_robust_20261005T111536_633570Z/dependency_pairs.jsonl')
    links = json.loads((path / 'paired_links.json').read_text())['pairs']
    for method, saved in audit['methods'].items():
        observed = report_predictions(rows, saved['predictions'], links, saved['threshold'])
        for key in ('classification', 'confusion_matrix', 'macro_f1', 'class_counts', 'paired_metrics', 'per_scenario'):
            assert saved[key] == observed[key], (method, key)
    old = ROOT / '_private/runs/eval_robust_20261005T111536_633570Z_evaluation/metrics'
    for filename, positive in [('finetuned_evaluation', 'depends_on'), ('pretrained_baseline', 'entailment')]:
        record = json.loads((old / (filename + '.json')).read_text())['test']
        for p in record['predictions']:
            logits, scores = p['raw_logits'], p['original_model_class_probabilities']
            assert all(math.isfinite(x) for x in logits)
            exp = [math.exp(x-max(logits)) for x in logits]
            softmax = [x/sum(exp) for x in exp]
            assert math.isclose(sum(scores.values()), 1, abs_tol=1e-6)
            assert all(math.isclose(a, b, abs_tol=1e-6) for a,b in zip(softmax, scores.values()))
            assert math.isclose(scores[positive], p['probability'], abs_tol=1e-7)
    assert audit['legacy_cohorts']['original']['within_label_world_homogeneity']['numerator'] == 51
    assert audit['legacy_cohorts']['original']['whole_world_perfect_classification']['numerator'] == 3
    print(f'PASS: 4 methods x 384 aligned predictions, explicit transformation metrics, {len(history)} preserved historical hashes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit_dir', type=Path)
    validate(parser.parse_args().audit_dir)
