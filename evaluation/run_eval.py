# Evaluation harness with explainability statistics

import sys
import os

# Ensure project root is on path
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import csv
from typing import List

from app.pipeline import DetectionPipeline
from app.decision import make_decision
from evaluation.metrics import compute_metrics, measure_latency


def load_dataset(path: str) -> List[dict]:
    """
    Load CSV dataset with columns: text,label
    label: 1 = injection, 0 = benign
    """
    samples = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({
                "text": row["text"],
                "label": int(row["label"])
            })
    return samples


def run_evaluation(dataset_path: str):
    pipeline = DetectionPipeline()

    y_true = []
    y_pred = []
    latencies = []

    rule_blocks = 0
    ml_blocks = 0

    samples = load_dataset(dataset_path)

    for sample in samples:
        text = sample["text"]
        true_label = sample["label"]

        latency, pipeline_output = measure_latency(pipeline.run, text)
        decision = make_decision(pipeline_output)

        pred_label = 1 if decision["decision"] == "BLOCK" else 0

        if pred_label == 1:
            if decision.get("layer") == "rules":
                rule_blocks += 1
            elif decision.get("layer") == "classifier":
                ml_blocks += 1

        y_true.append(true_label)
        y_pred.append(pred_label)
        latencies.append(latency)

    metrics = compute_metrics(y_true, y_pred)

    metrics["average_latency_ms"] = sum(latencies) / len(latencies)
    metrics["blocked_by_rules"] = rule_blocks
    metrics["blocked_by_classifier"] = ml_blocks
    metrics["total_samples"] = len(samples)

    return metrics


if __name__ == "__main__":
    results = run_evaluation("training/dataset.csv")

    print("Evaluation results:")
    for k, v in results.items():
        print(f"{k}: {v}")
