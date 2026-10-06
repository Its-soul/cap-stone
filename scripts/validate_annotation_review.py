"""Validate saved-evidence review and frozen v3 proposal without any model load."""
import argparse
import csv
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cert_recovery.annotation_review import (collect_reviews, summarize_reviews, transformation_review,
    ambiguity_queue, validate_sidecar, V2, STRESS)
from cert_recovery.contrast_data import generate_collection, generate_v3_candidates, audit_candidates
from cert_recovery.data_pipeline import read_jsonl
from cert_recovery.pinned_smoke import sha256_file


def validate(path):
    path = Path(path).resolve()
    def blocked(*args, **kwargs):
        raise AssertionError('Annotation validation prohibits network connections')
    socket.socket.connect = socket.socket.connect_ex = blocked
    load = lambda p: json.loads(p.read_text(encoding='utf-8'))
    history = load(path/'historical_hashes.json')
    for filename, digest in history.items():
        assert sha256_file(ROOT/filename) == digest, filename
    phase = load(path/'phase_manifest.json')
    for filename, digest in phase['prepared_artifact_hashes'].items():
        assert sha256_file(path/filename) == digest, filename
    for filename, digest in phase['preparation_source_hashes'].items():
        assert sha256_file(ROOT/filename) == digest, filename
    assert phase['status'] == 'FROZEN_PREPARATION_ONLY_NO_MODEL_EXECUTION'
    records, targeted, sources = collect_reviews(ROOT)
    review = load(path/'review.json')
    assert review['records'] == records and review['summary'] == summarize_reviews(records)
    assert review['source_files_sha256'] == sources
    assert review['targeted_review_ids'] == [r['review_id'] for r in targeted]
    assert len(records) == 544 and len(targeted) == 107
    assert validate_sidecar(path/'review.json', path/'corrections.json') == 144
    assert load(path/'historical_transformations.json') == transformation_review(ROOT)
    queue = load(path/'ambiguity_queue.json')
    assert queue['examples'] == ambiguity_queue() and all(r['label'] is None for r in queue['examples'])
    assert queue['historical_review_ids'] == [r['review_id'] for r in records if not r['text_sufficient_without_scope_assumptions']]
    contrast_rows, contrast_links = generate_collection()
    historical = read_jsonl(ROOT/V2/'dependency_pairs.jsonl') + read_jsonl(ROOT/STRESS)
    contrasts = load(path/'contrasts.json')
    assert contrasts['rows'] == contrast_rows and contrasts['transformations'] == contrast_links
    assert contrasts['audit'] == audit_candidates(contrast_rows, contrast_links, historical)
    rows, links, assignments = generate_v3_candidates()
    proposal = path/'proposal'
    for split, count in [('train', 256), ('validation', 64), ('future_final', 128)]:
        saved = read_jsonl(proposal/(split+'.jsonl'))
        assert saved == [r for r in rows if r['split'] == split] and len(saved) == count
        assert all('prediction' not in r and not r['human_reviewed'] for r in saved)
    assert load(proposal/'transformations.json')['links'] == links
    assert load(proposal/'leakage_audit.json') == audit_candidates(rows, links, historical)
    protocol = load(proposal/'protocol.json')
    assert protocol['group_assignments'] == assignments
    assert all(value is False for value in protocol['execution'].values())
    assert protocol['status'] == 'FROZEN_PROPOSAL_NOT_EXECUTED_PENDING_MENTOR'
    for split, digest in protocol['dataset_file_sha256'].items():
        assert sha256_file(proposal/(split+'.jsonl')) == digest
    assert sha256_file(proposal/'transformations.json') == protocol['transformation_file_sha256']
    assert sha256_file(proposal/'leakage_audit.json') == protocol['leakage_audit_file_sha256']
    v2_protocol = load(ROOT/V2/'protocol.json')
    assert protocol['controls']['starting_model'] == v2_protocol['model']
    assert protocol['controls']['training'] == v2_protocol['training']
    assert protocol['controls']['training_rows'] == 256 and protocol['controls']['optimizer_steps_budget'] == 256
    assert 'category recall' in protocol['acceptance_criteria']['metric_definitions']
    freeze = load(proposal/'freeze_manifest.json')
    for filename, digest in freeze['file_hashes'].items():
        assert sha256_file(proposal/filename) == digest
    assert freeze['protocol_sha256'] == sha256_file(proposal/'protocol.json')
    for filename, count in [('mentor_contrasts.csv', 64), ('mentor_errors.csv', 107)]:
        with (path/filename).open(encoding='utf-8', newline='') as stream:
            sheet = list(csv.DictReader(stream))
        assert len(sheet) == count
        assert all(not r[key] for r in sheet for key in ['mentor_name', 'review_date', 'decision', 'comments'])
    page = (path/'report.html').read_text(encoding='utf-8')
    assert 'PRELIMINARY / EXPERIMENTAL' in page and 'NO training, inference' in page
    assert '<script' not in page and 'href="http' not in page and 'src="http' not in page
    print(f'PASS: 544 saved reviews, 107 priority rows, 144 hash-bound proposals, 480 historical pair reviews; '
          f'64 provisional contrasts; frozen 256/64/128 data, 504 transformations; '
          f'{len(history)} unchanged historical artifacts; disabled execution, no future predictions')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('review_dir', type=Path)
    validate(parser.parse_args().review_dir)
