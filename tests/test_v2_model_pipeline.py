"""V2 logic with synthetic tensors; no checkpoint or model download is required."""
from contextlib import contextmanager
from copy import deepcopy
import math
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cert_recovery.data_pipeline import clean_rows, normalize_text
from cert_recovery.model_pipeline import _infer, _prediction_report, select_threshold


@pytest.fixture
def pair_rows():
    return [
        {"pair_id": "support", "world_id": "world", "premise": "D0 is not zero.",
         "hypothesis": "Q reads D0 only.", "label": 1, "source": "synthetic fixture"},
        {"pair_id": "excluded", "world_id": "world", "premise": "D2 is 17.",
         "hypothesis": "Q reads D0 only.", "label": 0, "source": "synthetic fixture"},
    ]


def test_v2_cleaning_preserves_scope_numbers_negation_and_unicode(pair_rows):
    rows = deepcopy(pair_rows)
    rows[0]["premise"] = "  Cafe\u0301  is not -17.\nNo other sources count. "
    original = deepcopy(rows)
    cleaned, audit = clean_rows(rows)
    assert cleaned[0]["premise"] == "Café is not -17. No other sources count."
    assert cleaned[0]["hypothesis"] == rows[0]["hypothesis"]
    assert [r["label"] for r in cleaned] == [1, 0]
    assert audit == {"raw_rows": 2, "clean_rows": 2, "removed_duplicates": 0}
    assert rows == original


@pytest.mark.parametrize("text", [None, 42, "", " \n\t "])
def test_v2_cleaning_rejects_nontext_or_empty_inputs(text):
    with pytest.raises(ValueError):
        normalize_text(text)


def test_v2_cleaning_rejects_contradictory_normalized_pairs(pair_rows):
    first = pair_rows[0]
    duplicate = dict(first, pair_id="contradiction", label=0, premise=" D0  is not zero. ")
    with pytest.raises(ValueError, match="Contradictory labels"):
        clean_rows([first, duplicate])


@pytest.mark.parametrize("mutation", ["missing_hypothesis", "boolean_label", "duplicate_id"])
def test_v2_cleaning_rejects_malformed_rows(pair_rows, mutation):
    rows = deepcopy(pair_rows)
    if mutation == "missing_hypothesis":
        del rows[0]["hypothesis"]
    elif mutation == "boolean_label":
        rows[0]["label"] = True
    else:
        rows[1]["pair_id"] = rows[0]["pair_id"]
    with pytest.raises(ValueError):
        clean_rows(rows)


def test_v2_report_uses_frozen_threshold_and_preserves_raw_outputs(pair_rows):
    before = deepcopy(pair_rows)
    raw = [
        {"raw_logits": [0.0, -1.38629436112],
         "original_model_class_probabilities": {"independent": .8, "depends_on": .2}},
        {"raw_logits": [-4.59511985013, 0.0],
         "original_model_class_probabilities": {"independent": .01, "depends_on": .99}},
    ]
    report = _prediction_report(pair_rows, [.2, .99], .2, 10, raw)
    assert [r["prediction"] for r in report["predictions"]] == [1, 1]
    assert report["predictions"][0]["class_probabilities"] == [.8, .2]
    assert report["predictions"][0]["predicted_label_probability"] == .2
    assert report["confusion_matrix"] == [[0, 1], [0, 1]]
    assert report["macro_f1"] == pytest.approx(1 / 3)
    assert report["false_negative_dependencies"] == 0
    assert [r["pair_id"] for r in report["examples"]["high_confidence_incorrect"]] == ["excluded"]
    for prediction, captured in zip(report["predictions"], raw):
        assert prediction["raw_logits"] == captured["raw_logits"]
        assert prediction["original_model_class_probabilities"] == captured["original_model_class_probabilities"]
    assert pair_rows == before


@pytest.mark.parametrize("probability", [float("nan"), float("inf"), -.01, 1.01])
def test_v2_report_rejects_invalid_probabilities(pair_rows, probability):
    with pytest.raises(ValueError, match="Probabilities must be finite"):
        _prediction_report(pair_rows, [probability, .5], .2, 10)


@pytest.mark.parametrize("probabilities,raw", [([.5], None), ([.5, .5], [{}])])
def test_v2_report_rejects_misaligned_outputs(pair_rows, probabilities, raw):
    with pytest.raises(ValueError, match="align with"):
        _prediction_report(pair_rows, probabilities, .2, 10, raw)


def test_v2_threshold_selection_prefers_recall_then_smallest_candidate():
    # Equal positive F1: threshold .2 has better recall than .5.
    assert select_threshold([1, 0, 0, 1], [.3, .3, .3, .8], [.5, .2]) == .2
    # Identical predictions: smallest threshold wins, regardless of candidate order.
    assert select_threshold([0, 1], [.1, .9], [.8, .5, .2]) == .2


@pytest.mark.parametrize("labels,probabilities,candidates", [
    ([], [], [.2]), ([1], [], [.2]), ([1], [.5], []),
    ([1], [.5], [-.1]), ([1], [.5], [float("nan")]),
])
def test_v2_threshold_selection_rejects_invalid_validation_data(labels, probabilities, candidates):
    with pytest.raises(ValueError):
        select_threshold(labels, probabilities, candidates)


class SyntheticTensor:
    def __init__(self, values):
        self.values = values
        self.devices = []

    def to(self, device):
        self.devices.append(device)
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values

    def softmax(self, dim):
        assert dim == -1
        probabilities = []
        for row in self.values:
            shifted = [math.exp(value - max(row)) for value in row]
            probabilities.append([value / sum(shifted) for value in shifted])
        return SyntheticTensor(probabilities)


@pytest.fixture
def inference_backend(monkeypatch):
    state = {"inside_inference": False}

    @contextmanager
    def inference_mode():
        state["inside_inference"] = True
        try:
            yield
        finally:
            state["inside_inference"] = False

    torch = SimpleNamespace(device=lambda name: name, cuda=SimpleNamespace(is_available=lambda: False),
                            inference_mode=inference_mode)
    monkeypatch.setitem(sys.modules, "torch", torch)
    token_batches = []

    def tokenize(premises, hypotheses, **kwargs):
        batch = {"input_ids": SyntheticTensor(premises)}
        token_batches.append((premises, hypotheses, kwargs, batch))
        return batch

    model = Mock()
    model.config = SimpleNamespace(id2label={0: "independent", 1: "depends_on"})
    model.side_effect = lambda **tokens: SimpleNamespace(logits=SyntheticTensor(
        [[0.0, math.log(3.0)] for _ in tokens["input_ids"].values]))
    return state, Mock(side_effect=tokenize), model, token_batches


def test_v2_inference_batches_complete_inputs_and_captures_binary_outputs(pair_rows, inference_backend):
    state, tokenizer, model, batches = inference_backend
    config = {"model": {"inference_batch_size": 1, "max_length": 128}}
    captured = []
    original_side_effect = model.side_effect

    def forward(**tokens):
        assert state["inside_inference"]
        return original_side_effect(**tokens)

    model.side_effect = forward
    probabilities = _infer(config, pair_rows, model, tokenizer, 1, captured)
    assert probabilities == pytest.approx([.75, .75])
    assert [batch[0] for batch in batches] == [[r["premise"]] for r in pair_rows]
    assert [batch[1] for batch in batches] == [[r["hypothesis"]] for r in pair_rows]
    assert all(batch[2] == {"padding": True, "truncation": True, "max_length": 128, "return_tensors": "pt"}
               for batch in batches)
    assert all(batch[3]["input_ids"].devices == ["cpu"] for batch in batches)
    model.eval.assert_called_once_with()
    model.to.assert_called_once_with("cpu")
    assert model.call_count == 2
    assert not state["inside_inference"]
    assert len(captured) == 2
    assert captured[0]["raw_logits"] == pytest.approx([0, math.log(3)])
    assert captured[0]["original_model_class_probabilities"] == pytest.approx({"independent": .25, "depends_on": .75})


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_v2_inference_rejects_invalid_batch_size(pair_rows, inference_backend, batch_size):
    _, tokenizer, model, _ = inference_backend
    with pytest.raises(ValueError, match="batch size must be positive"):
        _infer({"model": {"inference_batch_size": batch_size, "max_length": 128}}, pair_rows, model, tokenizer, 1)
    tokenizer.assert_not_called()
    assert model.call_count == 0


def test_v2_inference_propagates_model_failure_and_leaves_inference_mode(pair_rows, inference_backend):
    state, tokenizer, model, _ = inference_backend
    model.side_effect = RuntimeError("synthetic forward failure")
    capture = []
    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        _infer({"model": {"inference_batch_size": 2, "max_length": 128}}, pair_rows, model, tokenizer, 1, capture)
    assert capture == []
    assert not state["inside_inference"]
