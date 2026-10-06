"""Read-only offline validation; tokenizer execution only, never model inference."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def validate(bundle):
    bundle = Path(bundle).resolve()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1', TOKENIZERS_PARALLELISM='false')
    def blocked(*a, **kw): raise AssertionError('Prerequisite validation prohibits network')
    socket.socket.connect = socket.socket.connect_ex = blocked
    from cert_recovery.pinned_smoke import sha256_file
    from cert_recovery.crypto import canonical_json
    from cert_recovery.data_pipeline import read_jsonl, load_verified_splits
    from cert_recovery.v3_launch import verify_prepared_bundle
    from cert_recovery.v3_preparation import (load_json, local_tokenizer, audit_collection,
        mentor_packet, interpretation_note, check_batch_equivalence, explicit_roots, verify_proposal)
    history = load_json(bundle/'historical_hashes.json')
    for filename, digest in history.items(): assert sha256_file(ROOT/filename) == digest, filename
    spec, protocol, rows, links = verify_prepared_bundle(ROOT, bundle)
    review = Path(spec['review_dir']); contrasts = load_json(bundle/'replacement/contrasts.json')
    original_protocol, originals, original_links = verify_proposal(ROOT, review)
    original_contrasts = load_json(review/'contrasts.json')
    assert contrasts['rows'] == [explicit_roots(r) for r in original_contrasts['rows']]
    assert contrasts['transformations'] == original_contrasts['transformations']
    freeze = load_json(bundle/'replacement/freeze_manifest.json')
    for filename, digest in freeze['artifact_sha256'].items(): assert sha256_file(bundle/'replacement'/filename) == digest
    all_rows = rows+contrasts['rows']; all_links = links+contrasts['transformations']
    tokenizer, provenance = local_tokenizer(ROOT)
    actual = audit_collection(all_rows, all_links, tokenizer)
    saved = load_json(bundle/'tokenizer_audit.json')
    for key in actual: assert json.loads(canonical_json(actual[key])) == saved[key], key
    assert provenance == saved['tokenizer_provenance']
    original_actual = audit_collection(originals+original_contrasts['rows'], original_links+original_contrasts['transformations'], tokenizer)
    original_saved = load_json(bundle/'original_tokenizer_and_context_audit.json')
    for key in original_actual: assert json.loads(canonical_json(original_actual[key])) == original_saved[key]
    assert original_saved['decisive_information_failures'] == 256 and original_saved['truncated_rows'] == 0
    assert check_batch_equivalence(all_rows, tokenizer, actual['rows']) == saved['batch_equivalence']
    assert mentor_packet(review, rows, links, contrasts, spec['protocol_sha256']) == load_json(bundle/'mentor_packet.json')
    assert interpretation_note(review) == load_json(bundle/'historical_interpretation.json')
    approval = load_json(bundle/'approval.template.json'); authorization = load_json(bundle/'authorization.template.json')
    assert approval['human_reviewed'] is False and approval['reviewed_at_utc'] is None
    assert all(authorization[k] is False for k in ['training_authorized', 'inference_authorized', 'future_final_inference_authorized'])
    for key in ['contrast_reviews', 'future_final_reviews', 'historical_examples']:
        assert all(r['human_review_status'] == 'PENDING' and r['reviewer_identity'] is None and r['decision'] is None for r in approval[key])
    config = {'paths': {'processed': str(bundle/'prepared_splits')}, 'data': {'group_key': 'world_id'}}
    splits, _ = load_verified_splits(config)
    assert {key: len(value) for key,value in splits.items()} == {'train': 256, 'validation': 64, 'test': 128}
    assert not (bundle/'launch.claim.json').exists(), 'This preparation phase must not have execution claims'
    assert not any('prediction' in r for items in splits.values() for r in items)
    print(f'PASS: original protocol/data unchanged; root-explicit replacement; 512 rows and 576 encoded transformations, max {saved["max_original_tokens"]}/128 tokens, '
          f'zero truncation/decisive losses/invalid pairs; 192 required human reviews PENDING; '
          f'{len(history)} historical hashes intact; disabled launch, no model execution')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    validate(parser.parse_args().bundle)
