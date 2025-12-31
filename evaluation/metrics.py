# Evaluation metrics for prompt injection detector
import time
from typing import Callable, Tuple, List

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def compute_metrics(y_true: List[int], y_pred: List[int]) -> dict:
    """
    Compute standard classification metrics.
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate
    }


def measure_latency(func: Callable, *args, **kwargs) -> Tuple[float, any]:
    """
    Measure execution latency (in milliseconds) of a function.
    Returns latency and function result.
    """
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()

    latency_ms = (end - start) * 1000
    return latency_ms, result
