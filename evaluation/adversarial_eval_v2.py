#!/usr/bin/env python3
"""
FULL PIPELINE Adversarial Evaluation V2+ (100 Samples)
Uses the complete 5-layer DetectionPipeline with expanded ContextualGuard
"""

from pathlib import Path
import json
from app.pipeline import DetectionPipeline
from app.decision import make_decision

TEST_SET_PATH = Path("evaluation/adversarial_test_set_v2.json")

def run_full_pipeline_eval():
    print("=" * 80)
    print("FULL 5-LAYER PIPELINE EVALUATION (100 Samples)")
    print("=" * 80)

    # Load test set
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"]
    print(f"[OK] Loaded {len(samples)} adversarial prompts\n")

    # Load full pipeline
    pipeline = DetectionPipeline()

    blocked_count = 0
    results = []

    for idx, sample in enumerate(samples, 1):
        text = sample["hinglish"]
        pipeline_output = pipeline.run(text)
        decision = make_decision(pipeline_output)

        status = "BLOCKED" if decision["decision"] == "BLOCK" else "BYPASSED"
        if decision["decision"] == "BLOCK":
            blocked_count += 1

        layer = decision.get("layer", "unknown")
        reason = decision.get("reason", "unknown")

        print(f"[{idx:03d}] {status} | {sample['category']}")
        print(f"Prompt: {text[:80]}{'...' if len(text) > 80 else ''}")
        print(f"Blocked by: {layer} → {reason}")
        if "confidence" in decision:
            print(f"Confidence: {decision['confidence']:.4f}")
        print("-" * 60)

        results.append({
            "id": sample["id"],
            "category": sample["category"],
            "prompt": text,
            "actual": decision["decision"],
            "layer": layer,
            "reason": reason,
            "confidence": decision.get("confidence")
        })

    total = len(results)
    detection_rate = (blocked_count / total) * 100

    print("\n" + "=" * 80)
    print("FINAL SUMMARY - FULL PIPELINE")
    print("=" * 80)
    print(f"Total Prompts Tested     : {total}")
    print(f"Correctly BLOCKED        : {blocked_count}/{total} ({detection_rate:.1f}%)")
    print(f"Missed (BYPASSED)        : {total - blocked_count}/{total}")
    print("=" * 80)

    # Save results
    with open("evaluation/adversarial_results_full_pipeline.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Results saved to: evaluation/adversarial_results_full_pipeline.json")
    print("=" * 80)

if __name__ == "__main__":
    run_full_pipeline_eval()