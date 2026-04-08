# ============================================
# BASELINE PIPELINE RUN (ONNX + MINILM VERSION)
# ============================================

import sys
from sklearn.model_selection import train_test_split
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import time
import re
import yaml
import onnxruntime as rt
from sentence_transformers import SentenceTransformer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

from rules.instruction_rules import detect_instruction_override

# =========================
# CONFIG
# =========================
with open("config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

DATA_PATH = CONFIG["data"]["dataset_path"]
MODEL_PATH = CONFIG["classifier"]["model_path"]
EMBEDDER_NAME = CONFIG["embeddings"]["model_name"]
THRESHOLD = CONFIG["classifier"]["probability_threshold"]
TEST_SIZE = CONFIG["data"]["test_size"]
RANDOM_STATE = CONFIG["data"]["random_state"]
TEXT_COL = "text"
LABEL_COL = "label"

# =========================
# INITIALIZATION
# =========================
print("Initializing Production Pipeline...")
embedder = SentenceTransformer(EMBEDDER_NAME)
sess = rt.InferenceSession(MODEL_PATH)
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[1].name # Probabilities output

# =========================
# UTILITIES
# =========================
def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def rule_engine(text):
    result = detect_instruction_override(text)
    return result["override_detected"]

# =========================
# DATA PREP (For Evaluation)
# =========================
df = pd.read_csv(DATA_PATH, encoding="utf-8").dropna(subset=[TEXT_COL, LABEL_COL])
df[TEXT_COL] = df[TEXT_COL].apply(normalize_text)

# We use a 20% slice for final evaluation metrics
_, X_test, _, y_test = train_test_split(
    df[TEXT_COL], df[LABEL_COL], 
    test_size=TEST_SIZE, stratify=df[LABEL_COL], random_state=RANDOM_STATE
)

# =========================
# DECISION PIPELINE
# =========================
y_pred = []
rule_hits = 0
clf_hits = 0
latencies = []

print(f"Running Inference on {len(X_test)} samples...")

for text in X_test:
    start = time.time()
    text_norm = normalize_text(text)

    # Stage 1: Rule Engine
    if rule_engine(text_norm):
        y_pred.append(1)
        rule_hits += 1
    
    # Stage 2: ONNX AI Model
    else:
        # Generate Embeddings
        emb = embedder.encode([text_norm]).astype(np.float32)
        # Run ONNX Inference
        pred_onx = sess.run([label_name], {input_name: emb})[0]
        prob = pred_onx[0][1] # Probability of class 1

        if prob >= THRESHOLD:
            y_pred.append(1)
            clf_hits += 1
        else:
            y_pred.append(0)

    latencies.append(time.time() - start)

# =========================
# METRICS & REPORTING
# =========================
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("\n" + "="*30)
print("   FINAL PIPELINE REPORT")
print("="*30)
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"FPR:       {fp / (fp + tn):.4f}")
print("-" * 30)
print(f"Rule Detections: {rule_hits}")
print(f"AI Detections:   {clf_hits}")
print(f"Avg Latency:     {np.mean(latencies)*1000:.2f} ms")
print(f"Threshold:       {THRESHOLD}")
print("="*30)