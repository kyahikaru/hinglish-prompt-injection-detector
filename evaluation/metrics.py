# Evaluation metrics implementation will go here
import time
from typing import List, Dict


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    latency_ms: float = None
) -> Dict:
    """
    Compute evaluation metrics for prompt injection detection.
    Labels: 1 = injection, 0 = safe
    """

    assert len(y_true) == len(y_pred), "Label and prediction lengths must match"

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate": false_positive_rate
    }

    if latency_ms is not None:
        metrics["latency_ms"] = latency_ms

    return metrics


def measure_latency(func, *args, **kwargs) -> float:
    """
    Measure latency of a function call in milliseconds.
    """
    start = time.time()
    func(*args, **kwargs)
    end = time.time()
    return (end - start) * 1000.0
