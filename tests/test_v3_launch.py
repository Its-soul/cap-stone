"""All scores, review declarations and launch gates here are temporary TEST FIXTURES."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from cert_recovery.contrast_data import generate_v3_candidates
from cert_recovery.pinned_smoke import sha256_file
from cert_recovery.v3_launch import (validate_external_reviews, validate_authorization, initialize_attempt,
                                    future_metrics, acceptance, run_worker)


@pytest.fixture
def reviews():
    # Simulated records exist ONLY in this test; no actual mentor has reviewed the project.
    def item(name):
        return {'item_id': name, 'model_visible_input': {'premise': 'TEST', 'hypothesis': 'TEST'},
                'proposed_dependency_label': 0, 'reviewed_artifact_version': {'row_sha256': 'abc'},
                'reviewer_identity': None, 'decision': None, 'rationale': None, 'unresolved_ambiguity': None,
                'human_review_status': 'PENDING'}
    contract = {**item('contract'), 'reviewed_artifact_version': {'protocol_sha256': 'abc'}}
    packet = {'contract_review': contract, 'contrast_reviews': [item(str(i)) for i in range(64)],
              'future_final_reviews': [item(str(i)) for i in range(128)]}
    approval = copy.deepcopy(packet)
    for r in [approval['contract_review'], *approval['contrast_reviews'], *approval['future_final_reviews']]:
        r.update(reviewer_identity='SIMULATED UNIT TEST REVIEWER', decision='approve', rationale='TEST ONLY',
                 unresolved_ambiguity='', human_review_status='REVIEWED')
    approval.update(record_kind='EXTERNAL_HUMAN_REVIEW', human_reviewed=True, status='APPROVED',
        genuine_feedback_available=True, reviewed_at_utc='2026-10-06T00:00:00Z',
        reviewed_protocol_sha256='abc', reviewed_dataset_sha256={'train': 'abc'})
    return packet, approval, {'dataset_file_sha256': {'train': 'abc'}}


def test_complete_exact_review_coverage_validates_only_simulated_fixture(reviews):
    packet, approval, protocol = reviews
    result = validate_external_reviews(packet, approval, protocol)
    assert result['contrast_rows'] == 64 and result['future_final_rows'] == 128


@pytest.mark.parametrize('mutation', ['pending','missing','duplicate','changed_text','changed_label','changed_hash',
                                    'unresolved','no_identity','no_rationale','wrong_protocol','no_date'])
def test_pending_incomplete_or_rebound_review_is_rejected(reviews, mutation):
    packet, approval, protocol = reviews
    if mutation == 'pending': approval['human_reviewed'] = False
    elif mutation == 'missing': approval['future_final_reviews'].pop()
    elif mutation == 'duplicate': approval['contrast_reviews'].append(copy.deepcopy(approval['contrast_reviews'][0]))
    elif mutation == 'changed_text': approval['contrast_reviews'][0]['model_visible_input']['premise'] = 'OTHER'
    elif mutation == 'changed_label': approval['contrast_reviews'][0]['proposed_dependency_label'] = 1
    elif mutation == 'changed_hash': approval['future_final_reviews'][0]['reviewed_artifact_version']['row_sha256'] = 'other'
    elif mutation == 'unresolved': approval['contract_review']['unresolved_ambiguity'] = 'Unknown target'
    elif mutation == 'no_identity': approval['contrast_reviews'][0]['reviewer_identity'] = ''
    elif mutation == 'no_rationale': approval['contrast_reviews'][0]['rationale'] = ''
    elif mutation == 'wrong_protocol': approval['reviewed_protocol_sha256'] = 'different'
    else: approval['reviewed_at_utc'] = None
    with pytest.raises((ValueError, PermissionError)): validate_external_reviews(packet, approval, protocol)


@pytest.mark.parametrize('field', ['training_authorized','inference_authorized','future_final_inference_authorized','launch_spec_sha256'])
def test_execution_needs_every_permission_and_exact_launch_binding(tmp_path, field):
    path = tmp_path/'launch.json'; path.write_text('{}')
    auth = {'training_authorized': True, 'inference_authorized': True, 'future_final_inference_authorized': True,
            'launch_spec_sha256': sha256_file(path), 'max_training_attempts': 1,
            'authorized_by': 'SIMULATED TEST ONLY', 'authorized_at_utc': '2026-10-06T00:00:00Z'}
    auth[field] = False if field.endswith('authorized') else 'changed'
    with pytest.raises((ValueError, PermissionError)): validate_authorization(path, auth)


def test_launcher_disabled_before_imports_or_output_creation(tmp_path):
    script = Path(__file__).resolve().parents[1]/'scripts/run_candidate_v3.py'
    result = subprocess.run([sys.executable, str(script), '--bundle', str(tmp_path),
                             '--attempt-dir', str(tmp_path/'attempt')], capture_output=True, text=True)
    assert result.returncode != 0 and 'Disabled by default' in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_exclusive_claim_and_separate_attempt_preserve_prior_files(tmp_path, monkeypatch):
    import cert_recovery.v3_launch as module
    bundle = tmp_path/'bundle'; (bundle/'prepared_splits').mkdir(parents=True)
    (bundle/'prepared_splits/train.jsonl').write_text('TEST FIXTURE')
    (bundle/'launch_spec.json').write_text('{}')
    base = tmp_path/module.V2; base.mkdir(parents=True)
    (base/'config.json').write_text(json.dumps({'model': {}, 'training': {}, 'data': {}, 'paths': {}}))
    approval=tmp_path/'approval.json'; approval.write_text('TEST FIXTURE')
    auth=tmp_path/'auth.json'; auth.write_text('TEST FIXTURE')
    spec={'controls':{'starting_model':{},'training':{}}}
    monkeypatch.setattr(module,'validate_launch_gates',lambda *a:((spec,{},[],[]),{'test_fixture':True}))
    attempt=tmp_path/'first'
    initialize_attempt(tmp_path,bundle,attempt,approval,auth)
    original=(bundle/'launch.claim.json').read_bytes()
    with pytest.raises(FileExistsError): initialize_attempt(tmp_path,bundle,tmp_path/'second',approval,auth)
    assert not (tmp_path/'second').exists() and (bundle/'launch.claim.json').read_bytes()==original
    assert (attempt/'splits/train.jsonl').read_text()=='TEST FIXTURE'


def test_failed_training_cannot_open_future_final_or_import_models(tmp_path, monkeypatch):
    import cert_recovery.v3_launch as module
    bundle=tmp_path/'bundle';attempt=tmp_path/'attempt';bundle.mkdir();(attempt/'training').mkdir(parents=True)
    spec={'prepared_splits':{'file_sha256':{}}};config=attempt/'config.json';config.write_text('{}')
    approval=tmp_path/'approval';approval.write_text('TEST');auth=tmp_path/'auth';auth.write_text('TEST')
    (bundle/'launch_spec.json').write_text('{}')
    (bundle/'launch.claim.json').write_text(json.dumps({'attempt_directory':str(attempt.resolve())}))
    (attempt/'launch_binding.json').write_text(json.dumps({'authorization_sha256':sha256_file(auth),
        'approval_sha256':sha256_file(approval),'launch_spec_sha256':sha256_file(bundle/'launch_spec.json'),
        'config_sha256':sha256_file(config)}))
    (attempt/'training/finished.json').write_text(json.dumps({'status':'ERROR'}))
    monkeypatch.setattr(module,'validate_launch_gates',lambda *a:((spec,{},[],[]),{}))
    with pytest.raises(ValueError,match='Successful training required'):
        run_worker(tmp_path,bundle,attempt,'comparison',approval,auth)
    assert not (attempt/'comparison').exists()


def test_changed_selection_freeze_blocks_comparison_before_model_imports(tmp_path, monkeypatch):
    import cert_recovery.v3_launch as module
    bundle=tmp_path/'bundle';attempt=tmp_path/'attempt';bundle.mkdir();(attempt/'training').mkdir(parents=True)
    spec={'prepared_splits':{'file_sha256':{}}};config=attempt/'config.json';config.write_text('{}')
    approval=tmp_path/'approval';approval.write_text('TEST');auth=tmp_path/'auth';auth.write_text('TEST')
    (bundle/'launch_spec.json').write_text('{}')
    (bundle/'launch.claim.json').write_text(json.dumps({'attempt_directory':str(attempt.resolve())}))
    (attempt/'launch_binding.json').write_text(json.dumps({'authorization_sha256':sha256_file(auth),
        'approval_sha256':sha256_file(approval),'launch_spec_sha256':sha256_file(bundle/'launch_spec.json'),
        'config_sha256':sha256_file(config)}))
    selection=attempt/'selection_frozen.json';selection.write_text('{}')
    (attempt/'training/finished.json').write_text(json.dumps({'status':'SUCCESS','selection_frozen_sha256':sha256_file(selection)}))
    selection.write_text('{"tampered":true}')  # Temporary fixture, never a real selection.
    monkeypatch.setattr(module,'validate_launch_gates',lambda *a:((spec,{},[],[]),{}))
    with pytest.raises(ValueError,match='selection freeze changed'):
        run_worker(tmp_path,bundle,attempt,'comparison',approval,auth)
    assert not (attempt/'comparison').exists()


def test_frozen_metrics_and_joint_guards_using_synthetic_scores_only():
    rows,links,_=generate_v3_candidates()
    final=[r for r in rows if r['split']=='future_final'];ids={r['pair_id'] for r in final}
    selected=[l for l in links if l['parent_pair_id'] in ids]
    captures=[{'fixture_only':True,'raw_logits':[-1.,1.]} for r in final]
    perfect=future_metrics(final,[.98 if r['label'] else .02 for r in final],.2,captures,selected)
    constant=future_metrics(final,[.98]*len(final),.2,captures,selected)
    assert perfect['macro_f1']==1 and perfect['missed_dependency_rate']==0
    assert perfect['unchanged_value_positive_recall']['denominator']>0
    assert set(perfect['category_recall'])=={n+'_dependency_'+str(y) for n in ['entailment','contradiction','neutral'] for y in [0,1]}
    assert acceptance(perfect,constant)['all_pass']
    # Perfect v2 already has zero alarms: no 20pp gain can be claimed.
    assert not acceptance(perfect,perfect)['guards']['primary_false_alarm_reduction']
    assert not acceptance(constant,perfect)['all_pass']
