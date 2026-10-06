import copy
import pytest
from cert_recovery.controlled_data import generate_controlled, leakage_audit
from cert_recovery.evidence_audit import fit_baselines, baseline_predictions
from cert_recovery.data_pipeline import prepare_data, write_json


@pytest.fixture(scope='module')
def dataset():
    return generate_controlled()


def test_controlled_counts_and_derivations(dataset):
    rows, pairs, assignments = dataset
    audit = leakage_audit(rows, pairs)
    assert audit['split_rows'] == {'train': 256, 'validation': 64, 'test': 96}
    assert len(pairs) == 416 and len(assignments) == 18
    for r in rows:
        d = r['derivation']
        assert (d['intervention']['before'] != d['intervention']['after']) == bool(r['label'])
        assert d['intervention']['certificate_valid_after']
        assert d['intervention']['claim_version_after'] == (2 if r['label'] else 1)
        assert (d['intervention']['source'] in d['required_sources']) == bool(r['label'])
    for split, counts in audit['identifier_counts'].items():
        for name in {k.split(':')[0] for k in counts}:
            assert counts[name + ':0'] == counts[name + ':1']
        assert audit['positive_positions'][split]['0'] == audit['positive_positions'][split]['1']


def test_literal_entailment_is_separate_from_dependency(dataset):
    rows = [r for r in dataset[0] if r['scenario'] == 'nli_context']
    assert {r['label'] for r in rows} == {0, 1}
    assert all('entailment by repeated literal' in r['derivation']['expected_nli_reasoning'] for r in rows)


def test_identifier_fit_uses_training_only_and_reports_fallback(dataset):
    rows = dataset[0]
    fitted = fit_baselines([r for r in rows if r['split'] == 'train'])
    assert fitted['training_rows'] == 256
    assert fitted['global_probability'] == .5
    assert set(fitted['identifier_probabilities']) == {'Oak', 'Pine', 'Elm', 'Fir'}
    validation = [r for r in rows if r['split'] == 'validation']
    predictions, coverage = baseline_predictions(validation, fitted, True)
    assert coverage['coverage'] == .5
    assert all(p['probability'] == .5 for p in predictions)


@pytest.mark.parametrize('mutation', ['cross_split_variant', 'structural_leak', 'stress_overlap'])
def test_leakage_is_rejected(dataset, mutation):
    rows, pairs, _ = copy.deepcopy(dataset)
    stress = []
    if mutation == 'cross_split_variant':
        rows[0]['split'] = 'validation'
    elif mutation == 'structural_leak':
        victim = next(r for r in rows if r['split'] == 'validation')
        # Renaming world/aliases/numbers cannot disguise an entire shared template.
        victim['premise'], victim['hypothesis'] = rows[0]['premise'], rows[0]['hypothesis']
        pairs = []
    else:
        stress = [rows[0]]
    with pytest.raises(ValueError):
        leakage_audit(rows, pairs, stress)
