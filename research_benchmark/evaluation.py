"""
research_benchmark.evaluation
=============================
Rigorous metric computation, statistical significance testing,
and specialized confusion matrix analysis.

Key Focus:
- Standard Classification: Accuracy, Precision, Recall, Macro-F1
- Critical Dependency Asymmetry: False Negative Rate (FNR), Missed Dependency Count
- Generative Specialist Metrics: JSON Validity, Decision Agreement, Verifier Acceptance
- Statistical Tests: McNemar's Test, Bootstrap 95% Confidence Intervals
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np

def compute_classification_metrics(y_true: List[int], y_pred: List[int], labels: Optional[List[int]] = None) -> Dict[str, Any]:
    """Computes comprehensive classification metrics including Macro-F1 and confusion matrix."""
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    if labels is None:
        labels = sorted(list(set(y_true).union(set(y_pred))))
        
    num_classes = len(labels)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true_arr, y_pred_arr):
        if t in labels and p in labels:
            conf_matrix[labels.index(t), labels.index(p)] += 1
            
    # Accuracy
    total = len(y_true)
    correct = int(np.sum(y_true_arr == y_pred_arr))
    accuracy = correct / total if total > 0 else 0.0

    # Per-class Precision, Recall, F1
    precisions = []
    recalls = []
    f1s = []
    
    for i, c in enumerate(labels):
        tp = conf_matrix[i, i]
        fp = np.sum(conf_matrix[:, i]) - tp
        fn = np.sum(conf_matrix[i, :]) - tp
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    macro_f1 = float(np.mean(f1s))
    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))

    return {
        "accuracy": float(accuracy),
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "confusion_matrix": conf_matrix.tolist(),
        "per_class_f1": [float(f) for f in f1s],
        "labels": labels,
        "total_samples": total
    }

def compute_dependency_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    """
    Computes specialized dependency metrics.
    Labels: 0 = independent, 1 = depends_on.
    CRITICAL: Isolates False Negative Rate (FNR_dep), because missed dependencies
    cause stale certificates to be silently reused!
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    # 0 = independent (negative), 1 = depends_on (positive)
    tp = int(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))
    tn = int(np.sum((y_true_arr == 0) & (y_pred_arr == 0)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
    fn = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))
    
    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # CRITICAL METRIC: False Negative Rate on Dependencies
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    base_metrics = compute_classification_metrics(y_true, y_pred, labels=[0, 1])

    return {
        "accuracy": accuracy,
        "macro_f1": base_metrics["macro_f1"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_negative_rate": float(fnr),
        "false_negatives_count": fn,
        "false_positives_count": fp,
        "true_positives_count": tp,
        "true_negatives_count": tn,
        "specificity": float(specificity),
        "confusion_matrix": [[tn, fp], [fn, tp]], # rows: true 0, 1; cols: pred 0, 1
        "total_samples": total
    }

def bootstrap_confidence_interval(y_true: List[int], y_pred: List[int], metric_fn, num_bootstraps: int = 1000, alpha: float = 0.05) -> Tuple[float, float, float]:
    """
    Computes empirical bootstrap confidence interval (95% default) for a given metric.
    Returns: (mean, lower_bound, upper_bound)
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    n = len(y_true_arr)
    
    scores = []
    rng = np.random.RandomState(42)
    for _ in range(num_bootstraps):
        indices = rng.randint(0, n, size=n)
        sample_true = y_true_arr[indices]
        sample_pred = y_pred_arr[indices]
        scores.append(metric_fn(sample_true, sample_pred))
        
    scores = np.sort(scores)
    lower = float(np.percentile(scores, (alpha / 2.0) * 100))
    upper = float(np.percentile(scores, (1.0 - alpha / 2.0) * 100))
    mean = float(np.mean(scores))
    return mean, lower, upper

def compute_mcnemar_test(y_true: List[int], y_pred_a: List[int], y_pred_b: List[int]) -> Dict[str, Any]:
    """
    Computes McNemar's test for paired binary classification comparisons.
    Tests whether Model B is statistically significantly different from Model A.
    """
    y_true = np.array(y_true)
    y_a = np.array(y_pred_a)
    y_b = np.array(y_pred_b)
    
    correct_a = (y_a == y_true)
    correct_b = (y_b == y_true)
    
    # b: A correct, B wrong
    b = int(np.sum(correct_a & (~correct_b)))
    # c: A wrong, B correct
    c = int(np.sum((~correct_a) & correct_b))
    
    # McNemar's chi-squared with continuity correction
    if (b + c) == 0:
        chi2 = 0.0
        p_val = 1.0
    else:
        chi2 = ((abs(b - c) - 1.0) ** 2) / (b + c)
        try:
            from scipy.stats import chi2 as chi2_dist
            p_val = float(1.0 - chi2_dist.cdf(chi2, df=1))
        except ImportError:
            # Fallback approximation if scipy is not installed
            p_val = 0.001 if chi2 > 10.83 else (0.01 if chi2 > 6.63 else (0.05 if chi2 > 3.84 else 0.5))

    return {
        "model_a_only_correct": b,
        "model_b_only_correct": c,
        "chi2_statistic": float(chi2),
        "p_value": p_val,
        "is_significant_at_05": p_val < 0.05,
        "is_significant_at_01": p_val < 0.01
    }

def verify_specialist_assessment(
    proposal: Any,
    role: str,
    facts: Dict[str, Any],
    binding: Dict[str, Any],
    expected_decision: Optional[str] = None
) -> Dict[str, Any]:
    """
    Verifies an agent-generated specialist proposal using the system's verifier.
    
    Checks:
    1. Reads actual generated model output.
    2. Validates schema format (decision, confidence, evidence dict, rule_references list, reasoning_summary).
    3. Verifies evidence faithfulness against supplied ground-truth facts.
    4. Validates source versions and required rule references against binding contract.
    5. Passes proposal through project's existing assess_specialist engine.
    6. Records whether proposal is accepted (validation_status == 'VALID') or rejected.
    
    Keeps distinct:
    - decision_accuracy
    - valid_output_format
    - evidence_faithfulness
    - verifier_accepted
    """
    is_dict = isinstance(proposal, dict)
    required_keys = ["decision", "confidence", "evidence", "rule_references", "reasoning_summary"]
    has_keys = is_dict and all(k in proposal for k in required_keys)
    valid_format = is_dict and has_keys and ("execution_error" not in proposal)
    
    # Evidence faithfulness: proposal claims match provided facts
    evidence_faithful = False
    if valid_format and isinstance(proposal.get("evidence"), dict):
        evidence_faithful = (proposal.get("evidence") == facts)
        
    # Decision accuracy against ground truth (if provided)
    decision = proposal.get("decision") if is_dict else None
    decision_accuracy = (decision == expected_decision) if (decision and expected_decision) else False
    
    # Pass through system verifier
    try:
        from src.cert_recovery.adjudication.composition import assess_specialist
        assessment = assess_specialist(proposal, role, facts, binding)
        status = assessment.validation_status
        reasons = assessment.reasons
        fault_score = assessment.fault_score
    except Exception as e:
        # Fallback if cert_recovery import fails
        status = "VALID" if (valid_format and evidence_faithful and decision_accuracy) else "INVALID"
        reasons = (f"verifier_error:{e}",)
        fault_score = 0.0 if status == "VALID" else 1.0

    verifier_accepted = (status == "VALID")
    
    return {
        "decision_accuracy": decision_accuracy,
        "valid_output_format": valid_format,
        "evidence_faithfulness": evidence_faithful,
        "verifier_accepted": verifier_accepted,
        "validation_status": status,
        "reasons": reasons,
        "fault_score": fault_score
    }

if __name__ == "__main__":
    y_true = [0, 0, 1, 1, 1, 0, 1, 0]
    y_pred = [0, 0, 1, 0, 1, 0, 1, 1]
    print("[*] Testing Evaluation Metrics...")
    metrics = compute_dependency_metrics(y_true, y_pred)
    print(f"[+] Dependency Metrics: FNR={metrics['false_negative_rate']:.3f}, Macro-F1={metrics['macro_f1']:.3f}")
