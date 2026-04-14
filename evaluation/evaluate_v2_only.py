# evaluate_v2_only.py - FIXED for 250-sample adversarial set
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import onnxruntime as ort
from sklearn.metrics import classification_report

print("Loading adversarial prompts...")
with open("evaluation/adversarial_test_set_v2.json", "r", encoding="utf-8") as f:   # ← changed to v3
    data = json.load(f)

samples = data["samples"]
prompts = [sample["hinglish"] for sample in samples]
labels = [1] * len(prompts)          # all are injections

print(f"Loaded {len(prompts)} prompts successfully")

print("Encoding with MiniLM-L6-v2...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = embedder.encode(prompts).astype(np.float32)

print("Running ONLY V2 classifier...")
sess = ort.InferenceSession("models/final_classifier_v2.onnx")
input_name = sess.get_inputs()[0].name
output = sess.run(None, {input_name: emb})

if len(output) > 1:
    probs = output[1]
else:
    probs = output[0]

if isinstance(probs, list) and isinstance(probs[0], dict):
    probs = np.array([[d[0], d[1]] for d in probs])

probs = np.array(probs)
pred = (probs[:, 1] > 0.5).astype(int)

caught = (pred == 1).sum()

print("\n" + "="*80)
print(f"PURE V2 CLASSIFIER RESULTS — {len(prompts)} HARD ADVERSARIAL PROMPTS")
print("="*80)
print(classification_report(labels, pred, digits=4, zero_division=0))
print(f"\nV2 Hard Recall (label 1): {caught}/{len(prompts)} = {caught/len(prompts):.4f}")
print(f"V2 caught by itself: {caught}/{len(prompts)}")
