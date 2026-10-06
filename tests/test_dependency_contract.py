import pytest
from cert_recovery.dependency_contract import dependency_label, contract_witness
from cert_recovery.engine import CertificateSystem
from cert_recovery.models import ArtifactKind as K, VerificationResult


def test_transitive_declared_ancestry_not_minimal_causality():
    graph={'Q':['E'],'E':['Oak','Pine'],'Oak':[],'Pine':[],'Elm':[]}
    assert dependency_label('Oak','Q',graph)==1
    assert dependency_label('Elm','Q',graph)==0


def test_incomplete_absence_abstains_but_known_positive_survives():
    graph={'Q':['E'],'E':['Oak'],'Oak':[]}
    assert dependency_label('Pine','Q',graph,{'Q','Oak'}) is None
    assert dependency_label('Oak','Q',graph,{'Q','Oak'})==1


@pytest.mark.parametrize('graph,candidate,target',[({'Q':['Q']},'Oak','Q'),({'Q':[]},'Q','Q'),
    ({'Q':['Oak','Oak'],'Oak':[]},'Oak','Q'),({'Q':[]},'Oak','missing')])
def test_invalid_or_ambiguous_contract_rejected(graph,candidate,target):
    with pytest.raises(ValueError): dependency_label(candidate,target,graph)


@pytest.mark.parametrize('kind,state,changes',[('zero_read',{'Oak':2,'Pine':3},None),
    ('cancellation',{'Oak':2,'Pine':3},None),('redundant_or',{'Oak':1,'Pine':1},None),
    ('max_read',{'Oak':2,'Pine':5},None),('difference',{'Oak':2,'Pine':2},{'Oak':3,'Pine':3})])
def test_unchanged_value_never_removes_declared_support(kind,state,changes):
    record=contract_witness(kind,state,['Oak','Pine'],'Oak',changes)
    assert record['dependency_label']==1 and record['value_changed'] is False
    assert record['old_certificate_valid_after_change'] is False
    assert {'Q','C','Oak'}<=set(record['affected_ids'])
    assert record['old_certificate_status_after_recovery']=='SUPERSEDED'
    assert record['new_certificate_version']==2 and record['current_certificate_valid']
    assert record['declared_supports'][0]=={'artifact_id':'Oak','version':2}
    assert 'invalidation' in record['audit_events'] and record['audit_events'].count('certificate_issued')==2


def test_observed_but_undeclared_source_is_not_target_support():
    record=contract_witness('sum',{'Oak':2,'Pine':3,'Elm':4},['Pine','Elm'],'Oak')
    assert record['dependency_label']==0 and not record['value_changed']
    assert record['old_certificate_valid_after_change']
    assert record['new_certificate_version']==1 and 'Q' not in record['affected_ids']


def test_contradiction_with_dependency_does_not_certify_false_statement():
    system=CertificateSystem();source=system.add_dependency('Oak',{'value':6})
    system.register_verifier('exact-reading',lambda q,ancestry:VerificationResult(
        q.payload['value']==system.artifact(source).payload['value'],'exact source reading'))
    q=system.add_artifact('FalseCheck',K.CLAIM,{'value':7},(source,))
    assert dependency_label('Oak','FalseCheck',{'FalseCheck':['Oak'],'Oak':[]})==1
    with pytest.raises(ValueError,match='Verification failed'):system.issue_certificate('C',q,'exact-reading')
    assert not any(n.kind==K.CERTIFICATE for n in system.current_artifacts())
    assert any(r['kind']=='verification_result' and not r['payload']['passed'] for r in system.audit_export())
