# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from app.pipeline import DetectionPipeline
from app.decision import make_decision

print("Full Pipeline Evaluation on 110 Held-out Adversarial Samples")
print("=" * 90)

with open("benchmarks/hinglish-stealth-110-heldout.json", encoding="utf-8") as f:
    data = json.load(f)

samples = data["samples"]
print(f"Loaded {len(samples)} held-out adversarial prompts.\n")

pipeline = DetectionPipeline()
blocked_count = 0

for idx, sample in enumerate(samples, 1):
    result = pipeline.run(sample["hinglish"])
    decision = make_decision(result)
    
    final_dec = decision.get("decision") or decision.get("final_decision", "ALLOW")
    if final_dec == "BLOCK":
        blocked_count += 1
    
    if idx % 20 == 0 or idx == len(samples):
        print(f"Processed {idx}/{len(samples)} samples...")

detection_rate = (blocked_count / len(samples)) * 100

print("\n" + "="*90)
print("FULL PIPELINE RESULTS — HELD-OUT ADVERSARIAL SET")
print("="*90)
print(f"Total samples          : {len(samples)}")
print(f"Correctly BLOCKED      : {blocked_count}/{len(samples)}")
print(f"Detection Rate         : {detection_rate:.2f}%")
print(f"Missed                 : {len(samples) - blocked_count}")
print("="*90)
