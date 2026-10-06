"""Offline classification and explicitly linked transformation measurements."""
import re
from collections import defaultdict

from .data_pipeline import normalize_text
from .model_pipeline import _prediction_report


def rename_text(text, mapping):
    if not mapping or len(set(mapping.values())) != len(mapping) or set(mapping) != set(mapping.values()):
        raise ValueError("Renaming must be a bijection on the declared alias set")
    pattern = r"\b(?:" + "|".join(re.escape(k) for k in sorted(mapping,key=len,reverse=True)) + r")\b"
    return re.sub(pattern,lambda match:mapping[match.group()],text)


def align_predictions(rows, predictions):
    indexed = {}
    for row in rows:
        if row["pair_id"] in indexed:
            raise ValueError("Duplicate input pair_id")
        indexed[row["pair_id"]] = row
    aligned = {}
    for p in predictions:
        key = p["pair_id"]
        if key in aligned:
            raise ValueError("Duplicate prediction pair_id")
        if key not in indexed:
            raise ValueError("Unknown prediction pair_id")
        row = indexed[key]
        for field in ("premise","hypothesis","label"):
            if p.get(field) != row[field]:
                raise ValueError("Prediction-to-input mismatch: " + key + ":" + field)
        if type(p.get("prediction")) is not int or p["prediction"] not in (0,1):
            raise ValueError("Binary prediction required")
        aligned[key] = p
    if set(aligned) != set(indexed):
        raise ValueError("Missing predictions")
    return indexed, aligned


def validate_pairs(rows, pairs):
    indexed = {r["pair_id"]:r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate input IDs")
    seen = set()
    member_sets = set()
    for pair in pairs:
        if pair["comparison_id"] in seen:
            raise ValueError("Duplicate comparison ID")
        seen.add(pair["comparison_id"])
        a_id,b_id = pair["members"]
        if a_id == b_id or a_id not in indexed or b_id not in indexed:
            raise ValueError("Missing/duplicated pair member")
        marker = (pair["kind"],frozenset((a_id,b_id)))
        if marker in member_sets:
            raise ValueError("Duplicated transformation pair")
        member_sets.add(marker)
        a,b = indexed[a_id],indexed[b_id]
        if [a["label"],b["label"]] != pair["expected_labels"]:
            raise ValueError("Mismatched expected labels")
        if pair["kind"] == "renaming":
            if a["label"] != b["label"]:
                raise ValueError("Renaming changed expected label")
            for field in ("premise","hypothesis"):
                if normalize_text(rename_text(a[field],pair["alias_map"])) != normalize_text(b[field]):
                    raise ValueError("Non-alias context changed in renaming")
        elif pair["kind"] == "role_change":
            if a["label"] == b["label"]:
                raise ValueError("Role change must flip expected label")
            if pair.get("controlled") and a["premise"] != b["premise"] and not pair.get("context_rule_in_premise"):
                raise ValueError("Controlled role change altered source facts")
            edits = pair["changed_fields"]
            observed = [field for field in ("premise","hypothesis") if a[field] != b[field]]
            if edits != observed:
                raise ValueError("Mismatched transformation fields")
        else:
            raise ValueError("Unknown transformation kind")
    return indexed


def paired_metrics(rows, predictions, pairs):
    indexed, aligned = align_predictions(rows,predictions)
    validate_pairs(rows,pairs)
    result = {}
    all_members = {}
    for kind in ("renaming","role_change"):
        selected = [p for p in pairs if p["kind"] == kind]
        members = set()
        counts = defaultdict(int)
        success = 0
        both_correct = 0
        for pair in selected:
            a,b = pair["members"]
            members.update((a,b))
            counts[a] += 1
            counts[b] += 1
            correct = aligned[a]["prediction"] == indexed[a]["label"] and aligned[b]["prediction"] == indexed[b]["label"]
            both_correct += correct
            success += aligned[a]["prediction"] == aligned[b]["prediction"] if kind == "renaming" else correct
        definition = ("equal predicted binary labels for an exact bijective alias-only transformation; consistency need not be correctness"
                      if kind == "renaming" else "both linked counterpart predictions match their opposite expected labels; uncontrolled historical pairs are flagged")
        result[kind] = {"numerator":success,"denominator":len(selected),"rate":success/len(selected) if selected else None,
                        "both_correct_numerator":both_correct,"definition":definition,"unique_rows":len(members),
                        "members_in_multiple_pairs":sum(v>1 for v in counts.values()),
                        "controlled_pairs":sum(bool(p.get("controlled")) for p in selected),
                        "comparison_ids":[p["comparison_id"] for p in selected]}
        all_members[kind] = members
    result["coverage"] = {"total_rows":len(rows),"rows_in_both_test_types":len(all_members["renaming"] & all_members["role_change"]),
                           "association":"explicit comparison IDs, member pair IDs, alias maps, changed fields and expected label relationships"}
    return result


def report_predictions(rows, predictions, pairs, threshold, bins=10):
    _, aligned = align_predictions(rows,predictions)
    probs = [aligned[r["pair_id"]]["probability"] for r in rows]
    report = _prediction_report(rows,probs,threshold,bins)
    if any(report["predictions"][i]["prediction"] != aligned[r["pair_id"]]["prediction"] for i,r in enumerate(rows)):
        raise ValueError("Saved predictions disagree with frozen threshold")
    report["paired_metrics"] = paired_metrics(rows,predictions,pairs)
    report["per_scenario"] = {}
    for scenario in sorted({r.get("scenario") or r.get("template_family", "unknown") for r in rows}):
        subset = [r for r in rows if (r.get("scenario") or r.get("template_family", "unknown")) == scenario]
        subset_ids = {r["pair_id"] for r in subset}
        scoped_pairs = [p for p in pairs if set(p["members"]) <= subset_ids]
        scoped = _prediction_report(subset,[aligned[r["pair_id"]]["probability"] for r in subset],threshold,bins)
        report["per_scenario"][scenario] = {"classification":scoped["classification"],"macro_f1":scoped["macro_f1"],
            "confusion_matrix":scoped["confusion_matrix"],"class_counts":scoped["class_counts"],
            "paired_metrics":paired_metrics(subset,[aligned[r["pair_id"]] for r in subset],scoped_pairs),
            "high_confidence_errors":scoped["examples"]["high_confidence_incorrect"]}
    return report
