import copy
import pytest
from cert_recovery.contrast_data import (generate_collection,generate_v3_candidates,audit_candidates,
    validate_transformations,transformation_metrics,render,nli_label)


@pytest.fixture(scope='module')
def collection(): return generate_collection()


def test_all_six_categories_and_provisional_review(collection):
    rows,links=collection
    assert len(rows)==64 and len(links)==72
    assert {(r['nli_label'],r['label']) for r in rows}=={(n,y) for n in ['entailment','contradiction','neutral'] for y in [0,1]}
    assert all(not r['human_reviewed'] and r['annotation_status'].endswith('PENDING_HUMAN_REVIEW') for r in rows)
    assert {l['kind'] for l in links}=={'renaming','support_rule','lexical_similarity','facts_reorder','irrelevant_context'}


def test_support_changes_keep_facts_check_and_one_cofactor(collection):
    rows,links=collection;index={r['pair_id']:r for r in rows}
    for link in links:
        a,b=index[link['parent_pair_id']],index[link['child_pair_id']]
        if link['kind']=='support_rule':
            assert len(set(a['fields']['supports']) & set(b['fields']['supports']))==1
            assert a['fields']['state']==b['fields']['state'] and a['fields']['check']==b['fields']['check']
            assert (a['label'],b['label'])==(1,0) and a['nli_label']==b['nli_label']


@pytest.mark.parametrize('mutation',['facts','rule','nli','label','text','group','missing','duplicate','factor','human'])
def test_invalid_transformations_annotations_and_claims_rejected(collection,mutation):
    rows,links=copy.deepcopy(collection)
    if mutation=='facts':rows[1]['fields']['state']['Pine']=99
    elif mutation=='rule':rows[1]['fields']['rule_kind']='zero_read'
    elif mutation=='nli':rows[0]['nli_label']='neutral'
    elif mutation=='label':rows[0]['label']=0
    elif mutation=='text':rows[0]['premise']+=' Different fact.'
    elif mutation=='group':rows[1]['split']='future_final'
    elif mutation=='missing':links[0]['child_pair_id']='absent'
    elif mutation=='duplicate':links.append(copy.deepcopy(links[0]))
    elif mutation=='factor':links[0]['declared_factor']='facts'
    else:rows[0]['human_reviewed']=True
    with pytest.raises(ValueError):validate_transformations(rows,links)


def test_counts_grouping_positions_and_declared_interpolation():
    rows,links,assignments=generate_v3_candidates();audit=audit_candidates(rows,links)
    assert {k:v['rows'] for k,v in audit['split_audits'].items()}=={'train':256,'validation':64,'future_final':128}
    assert len(assignments)==28
    assert any(x['shared_structural_pairs']>0 for x in audit['overlap_audit'])
    final=[r for r in rows if r['split']=='future_final' and r['evaluation_track']=='family_held_out_from_train_and_validation']
    assert {r['scenario'] for r in final}=={'both_valid','max_read'}
    assert len(final)==32


def test_duplicate_history_and_sibling_cross_split_rejected(collection):
    rows,links=collection
    with pytest.raises(ValueError,match='Historical'):audit_candidates(rows,links,[rows[0]])
    changed=copy.deepcopy(rows);changed[1]['split']='other'
    with pytest.raises(ValueError):audit_candidates(changed,links)


def test_paired_metrics_constant_vs_perfect_and_shuffled_order(collection):
    rows,links=collection
    perfect=[{**r,'prediction':r['label']} for r in rows]
    result=transformation_metrics(list(reversed(rows)),list(reversed(perfect)),list(reversed(links)))
    assert all(v['numerator']==v['denominator'] for v in result.values())
    constant=transformation_metrics(rows,[{**r,'prediction':1} for r in rows],links)
    assert constant['renaming']['numerator']==constant['renaming']['denominator']
    assert constant['renaming']['both_correct_numerator']*2==constant['renaming']['denominator']
    assert constant['support_rule']['numerator']==0


def test_hidden_state_cannot_establish_nli_truth(collection):
    fields=copy.deepcopy(collection[0][0]['fields'])
    fields['fact_order']=[s for s in fields['fact_order'] if s!=fields['check']['source']]
    assert fields['check']['source'] in fields['state']
    assert nli_label(fields)=='neutral'


@pytest.mark.parametrize('mutation',['witness','track','family'])
def test_false_witness_or_heldout_claim_is_rejected(collection,mutation):
    rows,links=copy.deepcopy(collection)
    if mutation=='witness': rows[0]['derivation']['intervention']['value_changed']=False
    elif mutation=='track': rows[0]['evaluation_track']='family_held_out_from_train_and_validation'
    else: rows[0]['template_family']='undocumented-family'
    with pytest.raises(ValueError): audit_candidates(rows,links)
