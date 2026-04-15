import json
from pathlib import Path
from collections import Counter
from app.pipeline import ContextualGuard   # direct import of guard class

data_path = Path("evaluation/hinglish-clean-500.json")
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

samples = data["samples"]
guard = ContextualGuard()
guard_fp = 0
rule_counter = Counter()

for sample in samples:
    triggered, rule_name, _, _ = guard.check_red_flags(sample["hinglish"])
    if triggered:
        guard_fp += 1
        rule_counter[rule_name] += 1

print(f"Contextual Guard alone FPR: {guard_fp}/{len(samples)} = {(guard_fp/len(samples)*100):.3f}%")
print("Rule breakdown:", dict(rule_counter))