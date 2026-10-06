"""Synthetic tokenizer fixtures only. Exact pinned audit is a separate offline command."""
import copy
import re
import zlib

import pytest

from cert_recovery.contrast_data import generate_collection
from cert_recovery.v3_preparation import (model_input, audit_row, audit_collection, audit_transformations,
    span_retained, explicit_roots, original_row, ROOT_CLAUSE)
from cert_recovery.dependency_contract import dependency_label


class Encoding(dict):
    def sequence_ids(self): return self['_sequence_ids']


class FixtureTokenizer:
    """Word-offset fixture, explicitly NOT the DeBERTa tokenizer or an inference result."""
    truncation_side = padding_side = 'right'
    fixture_only = True
    def __call__(self, premise, hypothesis, truncation=False, max_length=None, **kwargs):
        def words(text):
            return [(zlib.crc32(m.group().encode())+10, m.span()) for m in re.finditer(r'\w+|[^\w\s]', text)]
        a, b = words(premise), words(hypothesis)
        if truncation:
            while len(a)+len(b)+3 > max_length:
                (a if len(a)>len(b) else b).pop()
        ids = [1]+[t for t,_ in a]+[2]+[t for t,_ in b]+[2]
        return Encoding(input_ids=ids, attention_mask=[1]*len(ids),
            offset_mapping=[(0,0)]+[s for _,s in a]+[(0,0)]+[s for _,s in b]+[(0,0)],
            special_tokens_mask=[1]+[0]*len(a)+[1]+[0]*len(b)+[1],
            _sequence_ids=[None]+[0]*len(a)+[None]+[1]*len(b)+[None])


@pytest.fixture(scope='module')
def collection():
    rows,links=generate_collection()
    return [explicit_roots(r) for r in rows],links


def test_input_uses_full_normalized_pair_and_ignores_projection_metadata(collection):
    row = copy.deepcopy(collection[0][0]); expected = {k: row[k] for k in ['premise','hypothesis']}
    row['nli_projection'] = {'premise': 'DECOY', 'hypothesis': 'DECOY'}
    row['premise'] = '  '+row['premise'].replace(' Facts:', '\n Facts:')+' '
    assert model_input(row) == expected
    assert 'Q' in expected['premise'] and 'ONLY' in expected['premise']


def test_three_special_tokens_and_precise_retention_not_just_length(collection):
    result = audit_row(collection[0][0], FixtureTokenizer())
    assert result['special_token_count'] == 3 and not result['truncated']
    assert result['essential_information_retained']
    assert all(c['status'] in ['RETAINED','NOT_APPLICABLE'] for c in result['clauses'])


@pytest.mark.parametrize('missing', ['exhaustivity','scope','query','rule','target'])
def test_short_input_with_missing_decisive_clause_is_not_sufficient(collection, missing):
    row = copy.deepcopy(collection[0][0])
    text = {'exhaustivity': 'ONLY', 'scope': 'Observation check is separate from Q support.',
            'query': 'candidate: Oak', 'rule': 'their sum', 'target': 'Q in contrast_w0000'}[missing]
    row['premise'] = row['premise'].replace(text, '')
    result = audit_row(row, FixtureTokenizer())
    assert not result['truncated'] and not result['essential_information_retained']
    assert result['missing_information'] and not result['lost_information']


def test_truncation_loses_decisive_rule_source_and_observation_clauses(collection):
    result = audit_row(collection[0][0], FixtureTokenizer(), max_length=14)
    assert result['truncated'] and not result['essential_information_retained']
    assert any('queried_source' in field for field in result['lost_information'])
    assert any('observation' in field for field in result['lost_information'])
    assert result['retained_length_including_special_tokens'] == 14


def test_partial_subword_span_loss_is_detected():
    full = Encoding(input_ids=[11,12], offset_mapping=[(0,3),(3,7)], _sequence_ids=[0,0])
    kept = Encoding(input_ids=[11], offset_mapping=[(0,3)], _sequence_ids=[0])
    assert not span_retained(full, kept, 0, 0, 7)
    assert span_retained(full, kept, 0, 0, 3)


def test_encoded_opposite_label_collision_rejects_pair_despite_retained_text(collection):
    rows, links = collection
    audits = [audit_row(r, FixtureTokenizer()) for r in rows]
    role = next(l for l in links if l['kind'] == 'support_rule')
    indexed = {r['case_id']: r for r in audits}
    indexed[role['child_pair_id']]['input_ids'] = indexed[role['parent_pair_id']]['input_ids'][:]
    results = audit_transformations(rows, links, audits)
    target = next(r for r in results if r['transformation_id'] == role['transformation_id'])
    assert target['status'] == 'INVALID_ENCODED_PAIR' and 'opposite-label collision' in target['reasons'][0]


def test_one_truncated_member_invalidates_encoded_transformation(collection):
    rows, links = collection; audits = [audit_row(r, FixtureTokenizer()) for r in rows]
    audits[0]['essential_information_retained'] = False; audits[0]['lost_information'] = ['hypothesis:exhaustivity']
    results = audit_transformations(rows, links, audits)
    assert any(r['status']=='INVALID_ENCODED_PAIR' and any('decisive' in reason for reason in r['reasons']) for r in results)


def test_full_fixture_collection_and_failure_counts_are_separate(collection):
    rows, links = collection
    complete = audit_collection(rows, links, FixtureTokenizer())
    assert complete['status'] == 'PASS' and len(complete['transformations']) == 72
    truncated = audit_collection(rows, links, FixtureTokenizer(), max_length=14)
    assert truncated['status'] == 'REPAIR_REQUIRED'
    assert truncated['truncated_rows'] == truncated['decisive_information_failures'] == 64
    assert truncated['invalid_encoded_pairs'] == 72


def test_direct_read_list_does_not_prove_transitive_absence_without_root_contract():
    direct_roots={'Q':['Pine','Elm'],'Pine':[],'Elm':[],'Oak':[]}
    derived_read={'Q':['Pine','Elm'],'Pine':['Oak'],'Elm':[],'Oak':[]}
    assert dependency_label('Oak','Q',direct_roots)==0
    assert dependency_label('Oak','Q',derived_read)==1
    rows,links=generate_collection()
    audit=audit_collection(rows,links,FixtureTokenizer())
    assert audit['truncated_rows']==0
    assert audit['decisive_information_failures']==32
    assert all(any('source_ancestry_contract' in m for m in r['missing_information']) for r in audit['rows'] if r['missing_information'])


def test_root_clause_repair_preserves_label_rule_and_parent_hash(collection):
    rows,_=collection
    for row in rows:
        parent=original_row(row)
        assert row['label']==parent['label'] and row['fields']==parent['fields']
        assert row['premise'].count(ROOT_CLAUSE)==row['hypothesis'].count(ROOT_CLAUSE)==1
    bad=copy.deepcopy(rows[0]);bad['label']=0
    with pytest.raises(ValueError,match='beyond root wording'):original_row(bad)


def test_negative_loses_ancestry_exhaustivity_when_root_clause_is_removed(collection):
    row=copy.deepcopy(next(r for r in collection[0] if r['label']==0))
    row['hypothesis']=row['hypothesis'].replace(ROOT_CLAUSE,'')
    audit=audit_row(row,FixtureTokenizer())
    assert not audit['truncated'] and 'hypothesis:source_ancestry_contract' in audit['missing_information']
