import json
from pathlib import Path
from collections import Counter
from app.pipeline import DetectionPipeline   # frozen pipeline

print("🚀 Upgrade 3 — FPR Evaluation on Hinglish-Clean-200")
print("=" * 80)

# Load the clean test set you confirmed
data_path = Path("evaluation/hinglish-clean-500.json")
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

samples = data["samples"]
print(f"Loaded {len(samples)} realistic clean Hinglish samples.\n")

pipeline = DetectionPipeline()

full_fp_count = 0
guard_fp_count = 0
rule_fp_counter = Counter()
detailed_results = []

for idx, sample in enumerate(samples):
    text = sample["hinglish"]
    result = pipeline.run(text)                    # ← full frozen 5-layer pipeline
    
    # Full pipeline decision
    final_decision = result["decision"].get("final_decision", "UNKNOWN")
    is_full_fp = (final_decision == "BLOCK")
    
    # Contextual Guard isolation (from pipeline output)
    guard_info = result.get("contextual_guard", {})
    guard_triggered = guard_info.get("red_flag_triggered", False)
    rule_name = guard_info.get("rule_name")
    
    if is_full_fp:
        full_fp_count += 1
    if guard_triggered:
        guard_fp_count += 1
        if rule_name:
            rule_fp_counter[rule_name] += 1
    
    detailed_results.append({
        "id": sample["id"],
        "text": text[:120] + "..." if len(text) > 120 else text,
        "full_pipeline_decision": final_decision,
        "contextual_guard_triggered": guard_triggered,
        "triggered_rule": rule_name or "None",
        "is_false_positive": is_full_fp
    })
    
    if (idx + 1) % 50 == 0 or idx == len(samples) - 1:
        print(f"✅ Processed {idx+1}/{len(samples)} samples")

# === FPR CALCULATION ===
total = len(samples)
fpr_full = (full_fp_count / total) * 100
fpr_guard_only = (guard_fp_count / total) * 100

print("\n" + "="*80)
print("FINAL FPR RESULTS (Upgrade 3)")
print("="*80)
print(f"Total clean samples          : {total}")
print(f"False Positives — Full Pipeline (5 layers) : {full_fp_count} → {fpr_full:.3f}%")
print(f"False Positives — Contextual Guard alone   : {guard_fp_count} → {fpr_guard_only:.3f}%")

print("\nPer-Rule False Positive Breakdown (Contextual Guard):")
if rule_fp_counter:
    for rule, count in rule_fp_counter.most_common():
        print(f"   • {rule:<55} : {count} times")
else:
    print("   No rules triggered — perfect!")

# Save detailed results for paper
output_file = Path("evaluation/clean_fpr_results.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump({
        "total_samples": total,
        "fp_full_pipeline": full_fp_count,
        "fpr_full_pipeline_percent": round(fpr_full, 3),
        "fp_contextual_guard": guard_fp_count,
        "fpr_contextual_guard_percent": round(fpr_guard_only, 3),
        "rule_breakdown": dict(rule_fp_counter),
        "detailed_samples": detailed_results
    }, f, indent=2, ensure_ascii=False)

print(f"\n📁 Detailed results + breakdown saved to: {output_file}")
print("Final Table Ready.")