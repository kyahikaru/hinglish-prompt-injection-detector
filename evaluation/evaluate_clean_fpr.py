# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from pathlib import Path
from collections import Counter
from app.pipeline import DetectionPipeline

print("Upgrade 4 - FPR Evaluation on Hinglish-Clean-500")
print("=" * 80)

data_path = Path("benchmarks/hinglish-clean-500.json")
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

samples = data["samples"]
print(f"Loaded {len(samples)} realistic clean Hinglish samples.")

pipeline = DetectionPipeline()

full_fp_count = 0
guard_fp_count = 0
rule_fp_counter = Counter()

for idx, sample in enumerate(samples):
    text = sample["hinglish"]
    result = pipeline.run(text)
    
    decision_dict = result.get("decision", {})
    final_decision = decision_dict.get("decision") or decision_dict.get("final_decision", "UNKNOWN")
    is_full_fp = (final_decision == "BLOCK")
    
    guard_info = result.get("contextual_guard", {})
    guard_triggered = guard_info.get("red_flag_triggered", False)
    rule_name = guard_info.get("rule_name")
    
    if is_full_fp:
        full_fp_count += 1
    if guard_triggered:
        guard_fp_count += 1
        if rule_name:
            rule_fp_counter[rule_name] += 1
    
    if (idx + 1) % 100 == 0 or idx == len(samples) - 1:
        print(f"Processed {idx+1}/{len(samples)} samples")

total = len(samples)
fpr_full = (full_fp_count / total) * 100
fpr_guard_only = (guard_fp_count / total) * 100

print("\n" + "="*80)
print("FINAL FPR RESULTS (Upgrade 4)")
print("="*80)
print(f"Total clean samples          : {total}")
print(f"False Positives - Full Pipeline : {full_fp_count} -> {fpr_full:.3f}%")
print(f"False Positives - Contextual Guard alone : {guard_fp_count} -> {fpr_guard_only:.3f}%")

print("\nPer-Rule False Positive Breakdown:")
if rule_fp_counter:
    for rule, count in rule_fp_counter.most_common():
        print(f"   * {rule}: {count} times")
else:
    print("   No rules triggered - excellent!")

print(f"\nContextual Guard FPR: {fpr_guard_only:.3f}%")
print("Evaluation completed.")
