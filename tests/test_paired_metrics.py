import copy
import pytest
from cert_recovery.paired_metrics import paired_metrics


def fixture():
    rows = [{"pair_id":"a","premise":"Oak value 1.","hypothesis":"Sum uses Oak.","label":1},
            {"pair_id":"b","premise":"Pine value 1.","hypothesis":"Sum uses Pine.","label":1},
            {"pair_id":"c","premise":"Oak value 1.","hypothesis":"Sum uses Pine.","label":0},
            {"pair_id":"d","premise":"Pine value 1.","hypothesis":"Sum uses Oak.","label":0}]
    pairs = [{"comparison_id":"rename-pos","members":["a","b"],"kind":"renaming","expected_labels":[1,1],"alias_map":{"Oak":"Pine","Pine":"Oak"},"controlled":True},
             {"comparison_id":"rename-neg","members":["c","d"],"kind":"renaming","expected_labels":[0,0],"alias_map":{"Oak":"Pine","Pine":"Oak"},"controlled":True},
             {"comparison_id":"role","members":["a","c"],"kind":"role_change","expected_labels":[1,0],"changed_fields":["hypothesis"],"controlled":True}]
    return rows,pairs


def predictions(rows,labels):
    return [{**r,"prediction":y} for r,y in zip(rows,labels)]


def test_perfect_predictions_and_overlapping_coverage():
    rows,pairs=fixture()
    result=paired_metrics(rows,predictions(rows,[1,1,0,0]),pairs)
    assert result["renaming"]["numerator"] == result["renaming"]["denominator"] == 2
    assert result["role_change"]["numerator"] == 1
    assert result["renaming"]["unique_rows"] == 4
    assert result["coverage"]["rows_in_both_test_types"] == 2


def test_constant_predictions_consistent_not_role_correct():
    rows,pairs=fixture()
    result=paired_metrics(rows,predictions(rows,[1]*4),pairs)
    assert result["renaming"]["rate"] == 1
    assert result["renaming"]["both_correct_numerator"] == 1
    assert result["role_change"]["rate"] == 0


def test_consistently_wrong_renaming_is_not_correctness():
    rows,pairs=fixture()
    result=paired_metrics(rows,predictions(rows,[0,0,1,1]),pairs)
    assert result["renaming"]["rate"] == 1
    assert result["renaming"]["both_correct_numerator"] == 0
    assert result["role_change"]["rate"] == 0


@pytest.mark.parametrize("values,expected",[([1,0,0,1],1),([0,1,1,0],0),([1,1,1,1],0)])
def test_role_change_both_sides_required(values,expected):
    rows,pairs=fixture()
    assert paired_metrics(rows,predictions(rows,values),pairs)["role_change"]["numerator"] == expected


def test_shuffling_predictions_rows_and_pair_order_does_not_change_counts():
    rows,pairs=fixture()
    p=predictions(rows,[1,1,0,0])
    original=paired_metrics(rows,p,pairs)
    shuffled=paired_metrics(list(reversed(rows)),[p[2],p[0],p[3],p[1]],list(reversed(pairs)))
    for kind in ("renaming","role_change"):
        assert original[kind]["numerator"] == shuffled[kind]["numerator"]
        assert original[kind]["unique_rows"] == shuffled[kind]["unique_rows"]


@pytest.mark.parametrize("mode",["missing","duplicate","input_mismatch","unknown","bad_label","bad_pair","missing_pair","duplicate_pair","context_change"])
def test_invalid_members_or_alignment_rejected(mode):
    rows,pairs=fixture()
    p=predictions(rows,[1,1,0,0])
    if mode=="missing": p.pop()
    elif mode=="duplicate": p.append(copy.deepcopy(p[0]))
    elif mode=="input_mismatch": p[0]["premise"]="Changed source"
    elif mode=="unknown": p[0]["pair_id"]="absent"
    elif mode=="bad_label": pairs[0]["expected_labels"]=[0,0]
    elif mode=="bad_pair": pairs[0]["members"]=["a","a"]
    elif mode=="missing_pair": pairs[0]["members"]=["a","absent"]
    elif mode=="duplicate_pair": pairs.append({**pairs[0],"comparison_id":"duplicate-members"})
    elif mode=="context_change": rows[1]["premise"]="Pine value 2."; p=predictions(rows,[1,1,0,0])
    with pytest.raises(ValueError): paired_metrics(rows,p,pairs)


def test_no_implicit_world_pairs_or_probability_zip_truncation():
    from scripts.run_evaluation import compute_paired_metrics
    rows,pairs=fixture()
    assert compute_paired_metrics(rows,[.9]*4,.5)["status"].startswith("UNAVAILABLE")
    with pytest.raises(ValueError,match="Every probability"):
        compute_paired_metrics(rows,[.9],.5,pairs)
