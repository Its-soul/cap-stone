"""Synthetic saved-score fixtures only; no model or private artifact required."""
import copy
import json

import pytest

from cert_recovery.annotation_review import review_example, summarize_reviews, write_exclusive, validate_sidecar
from cert_recovery.crypto import sha256
from cert_recovery.pinned_smoke import sha256_file


def example(scenario='nli_context', label=0, prediction=1, probability=.97):
    candidate = 'Oak'
    supports = ['Pine', 'Elm'] if label == 0 else ['Oak', 'Pine']
    rule = 'sum the complete support list ' + ', '.join(supports)
    row = {'pair_id': 'synthetic-recorded-score-fixture', 'scenario': scenario, 'label': label,
           'premise': 'Q uses only Pine and Elm. Oak reads 6.',
           'hypothesis': 'Q uses only Pine and Elm. Oak reads 6.',
           'derivation': {'rule': rule, 'required_sources': supports,
                          'intervention': {'source': candidate}}}
    if scenario == 'nli_trap':
        row['premise'] = 'Service Oak is used for backup storage.'
        row['hypothesis'] = 'Oak is fully operational and holds identical data to the main storage.'
    if scenario == 'approval':
        row['premise'] = 'Oak was signed yesterday.'
        row['hypothesis'] = 'Release requires sign-off from Pine and Elm.'
    if scenario == 'arithmetic':
        row['premise'] = 'Oak has value 6.'
        row['hypothesis'] = 'The result is computed exactly from Pine and Elm.'
    predicted_probability = probability if prediction else 1 - probability
    p = {**row, 'prediction': prediction, 'probability': probability, 'correct': prediction == label,
         'predicted_label_probability': predicted_probability, 'raw_logits': [-1., 1.],
         'original_model_class_probabilities': {'independent': 1-probability, 'depends_on': probability}}
    return row, p


def test_scope_dependent_error_preserves_input_derivation_and_scores():
    row, p = example()
    r = review_example(row, p, p, .2, .2, 'validation')
    assert r['finding'] == 'SCOPE_AMBIGUITY_CONDITIONAL_MODEL_ERROR'
    assert not r['text_sufficient_without_scope_assumptions']
    assert r['input'] == {k: row[k] for k in ['premise', 'hypothesis']}
    assert r['original_row_sha256'] == sha256(row)
    assert r['original_derivation'] == row['derivation']
    assert r['predictions']['candidate_v2']['probability'] == .97
    assert r['predictions']['candidate_v2']['threshold'] == .2
    assert r['proposed_correction']['binary_label'] == 0
    assert not r['human_reviewed'] and not r['predictions']['candidate_v2']['calibrated']


@pytest.mark.parametrize('scenario', ['approval', 'nli_trap'])
def test_unknown_contract_does_not_invert_historical_label(scenario):
    row, p = example(scenario)
    r = review_example(row, p, p, .2, .2, 'stress')
    assert r['finding'] == 'UNDERDETERMINED_ANNOTATION'
    assert r['reviewed_dependency_label_under_explicit_scope'] is None
    assert r['proposed_correction']['binary_label'] is None
    assert r['original_label'] == 0


def test_complete_rule_failure_is_distinguished_from_scope_ambiguity():
    row, p = example('arithmetic')
    r = review_example(row, p, p, .2, .2, 'stress')
    assert r['finding'] == 'MODEL_ERROR_UNDER_EXPLICIT_PROVISIONAL_RULE'
    assert r['text_sufficient_without_scope_assumptions']
    assert r['reviewed_dependency_label_under_explicit_scope'] == 0


@pytest.mark.parametrize('mutation', ['input', 'ordering', 'threshold'])
def test_wrong_saved_input_id_or_threshold_rejected(mutation):
    row, p = example()
    if mutation == 'input': p['hypothesis'] += ' Other source.'
    elif mutation == 'ordering': p['pair_id'] = 'wrong-record'
    else: p['prediction'] = 0
    with pytest.raises(ValueError): review_example(row, p, p, .2, .2, 'validation')


def test_confidence_subset_is_not_counted_as_extra_errors():
    records = []
    for split in ['validation', 'final', 'stress']:
        row, p = example('nli_context' if split != 'stress' else 'nli_trap')
        records.append(review_example(row, p, p, .2, .2, split))
    counts = summarize_reviews(records)
    assert counts['unique_reviewed_rows'] == 3
    assert counts['independent_nli_context_errors'] == 2
    assert counts['validation_errors_at_least_point9'] == 1
    assert counts['findings']['UNDERDETERMINED_ANNOTATION'] == 1
    assert sum(counts['confidence_distributions']['validation']['candidate_v2']['all_counts']) == 1


@pytest.fixture
def sidecar(tmp_path):
    row, p = example()
    record = review_example(row, p, p, .2, .2, 'validation')
    review = tmp_path/'review.json'; corrections = tmp_path/'corrections.json'
    write_exclusive(review, {'records': [record]})
    proposal = {'review_id': record['review_id'], 'original_row_sha256': record['original_row_sha256'],
                'original_label': record['original_label'], 'proposed': record['proposed_correction'],
                'status': 'PENDING_HUMAN_REVIEW', 'human_reviewed': False}
    write_exclusive(corrections, {'original_review_sha256': sha256_file(review), 'proposals': [proposal]})
    return review, corrections


def test_sidecar_write_is_exclusive_and_does_not_change_bytes(sidecar):
    review, corrections = sidecar
    before = {p: p.read_bytes() for p in sidecar}
    assert validate_sidecar(review, corrections) == 1
    with pytest.raises(FileExistsError): write_exclusive(corrections, {'replacement': True})
    assert {p: p.read_bytes() for p in sidecar} == before


@pytest.mark.parametrize('mutation', ['label', 'hash', 'human', 'duplicate', 'review'])
def test_tampered_or_unbound_corrections_are_rejected(sidecar, mutation):
    review, corrections = sidecar
    data = json.loads(corrections.read_text())
    if mutation == 'label': data['proposals'][0]['original_label'] = 1
    elif mutation == 'hash': data['proposals'][0]['original_row_sha256'] = '0'*64
    elif mutation == 'human': data['proposals'][0]['human_reviewed'] = True
    elif mutation == 'duplicate': data['proposals'].append(copy.deepcopy(data['proposals'][0]))
    else: review.write_text(review.read_text() + ' ')
    # Deliberately damaged temporary fixture, never a historical or prepared sidecar.
    corrections.write_text(json.dumps(data))
    with pytest.raises(ValueError): validate_sidecar(review, corrections)
