#!/usr/bin/env python3
"""
FULL PIPELINE Adversarial Evaluation V2+ (100 Samples)
With clear breakdown: Pure V2 vs Contextual Guard + Rules
"""

from pathlib import Path
import json
from app.pipeline import DetectionPipeline
from app.decision import make_decision

TEST_SET_PATH = Path("benchmarks/hinglish-stealth-110-heldout.json")
RESULTS_PATH = Path("evaluation/adversarial_results_v3_pipeline.json")

def run_full_pipeline_eval():
    print("=" * 80)
    print("FULL 5-LAYER PIPELINE EVALUATION (100 Samples)")
    print("=" * 80)

    # Load test set
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"]
    print(f"[OK] Loaded {len(samples)} adversarial prompts\n")

    pipeline = DetectionPipeline()

    blocked_count = 0
    v2_classifier_blocked = 0
    contextual_guard_blocked = 0
    rule_blocked = 0
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

        # Count who actually blocked it
        if layer == "v2_classifier":
            v2_classifier_blocked += 1
        elif layer == "contextual_guard":
            contextual_guard_blocked += 1
        elif layer == "rule_engine":
            rule_blocked += 1

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
    print(f"Total Prompts Tested          : {total}")
    print(f"Pure V2 Classifier alone      : {v2_classifier_blocked}/100")
    print(f"Contextual Guard + Rules      : {contextual_guard_blocked + rule_blocked}/100")
    print(f"Correctly BLOCKED (Total)     : {blocked_count}/100 ({detection_rate:.1f}%)")
    print(f"Missed (BYPASSED)             : {total - blocked_count}/100")
    print("=" * 80)

    # Save detailed results
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {RESULTS_PATH}")
    print("=" * 80)

if __name__ == "__main__":
    run_full_pipeline_eval()